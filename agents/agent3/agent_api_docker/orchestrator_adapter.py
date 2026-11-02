"""
Adaptateur de l'Agent 3 (API + Docker) pour l'orchestrateur (contrat : common/schemas.py).

    REAL["agent3"] = "agent_api_docker.orchestrator_adapter:run"

Ce que fait l'adaptateur autour du pipeline de l'agent (loop.run_pipeline, inchangé) :
1. récupère le modèle promu par l'Agent 2 (mlflow_metadata.model_uri) et l'écrit en model/model.joblib dans une
   copie du projet, avec un petit module d'inférence (mlflow_predict.py) ; repli : le modèle du projet, entraîné
   au build comme avant (deployment_result.model_source = "project_artifact") ;
2. construit le rapport d'analyse v1.0 qu'attend l'agent, depuis project_spec, data_profile et le CSV : versions
   de Python et des librairies = celles qui ont entraîné le modèle (sinon le pickle ne se recharge pas) ;
3. lance le pipeline de l'agent : génération, validation, tests locaux, puis image Docker et tests dans le conteneur
   (corrections par le LLM si LLM_* est configuré dans agents/agent3/.env) ;
4. démarre le conteneur validé (nom mlops-<projet>, port AGENT3_PORT, 8000 par défaut) et attend /health :
   deployment_result.endpoint est l'API que l'Agent 4 surveille ;
5. pose l'alias « champion » sur la version MLflow déployée (mlflow_agent.orchestrator_adapter.promote).

Sans Docker (démon arrêté) : l'API est validée et testée en local, mais pas déployée (endpoint vide, WARNING).
Variables : AGENT3_DOCKER=0 (pas de Docker), AGENT3_PORT, AGENT3_NO_FIX=1 (pas de correction LLM),
AGENT3_PUSH=1 (publication sur Docker Hub, identifiants DOCKERHUB_* de agents/agent3/.env).

Publication : l'agent ne pousse jamais sans approbation. Dans le pipeline, l'approbation est la gate humaine
« champion_replacement » de l'orchestrateur (mise en production), donnée avant l'Agent 3.
En ligne de commande interactive (ORCH_INTERACTIVE=1, posé par python -m orchestrator), comme le CLI de l'agent :
destination affichée, choix du compte (Entrée = compte du .env, sinon un autre compte et son token, saisi sans écho
et jamais enregistré), puis confirmation du push.
Attention : un dépôt créé par le push est PUBLIC sur Docker Hub ; le créer d'abord en privé si besoin.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import pandas as pd

from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    CheckResult,
    DeploymentResult,
    ValidatorHandoff,
    read_output,
    write_output,
)

MODEL_FILE = "model/model.joblib"
PREDICT_MODULE = "mlflow_predict.py"
CONTAINER_PORT = 8000
IGNORE = [".git", "__pycache__/", "*.pyc", ".venv/", "venv/", "output/", "*.ipynb", ".env", "mlruns/",
          "test_api_generated.py", "junit.xml", "server.log"]

_PREDICT_SRC = '''"""Inférence du modèle enregistré dans MLflow (écrit par l'adaptateur de l'Agent 3, ne pas éditer)."""
import pandas as pd

# Le modèle de l'Agent 2 prédit l'indice de la classe (LabelEncoder, classes triées) : on rend l'étiquette d'origine.
CLASSES = {classes}


def predict(model, data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    out = model.predict(df).tolist()
    return [CLASSES[int(p)] for p in out] if CLASSES else out
'''


def _writable_copy(src: Path, dest: Path) -> Path:
    """Copie du projet validé (lecture seule) dans le workdir, modifiable."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", "*.pyc",
                                                              "output", "expected_reports"))
    for p in [dest, *dest.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)
    return dest


def _versions() -> dict[str, str]:
    """Versions de l'environnement qui a entraîné le modèle (l'orchestrateur et l'Agent 2 partagent la .venv)."""
    import joblib
    import numpy
    import scipy
    import sklearn

    return {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "joblib": joblib.__version__}


def _export_mlflow_model(meta, build: Path) -> None:
    """Charge le modèle depuis le registre MLflow et l'écrit en joblib : l'image n'a pas besoin de MLflow."""
    import joblib
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(meta.tracking_uri)
    model = mlflow.sklearn.load_model(meta.model_uri)
    (build / MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, build / MODEL_FILE)


def _field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_numeric_dtype(series):
        return "float"
    return "str"


def _input_schema(df: pd.DataFrame, features: list[str]) -> list[dict]:
    fields = []
    for name in features:
        s, kind = df[name], _field_type(df[name])
        f = {"name": name, "type": kind, "required": True}
        if kind in ("int", "float"):
            f.update(min=float(s.min()), max=float(s.max()))
        elif kind == "str" and s.nunique() <= 50:
            f["allowed_values"] = sorted(str(v) for v in s.dropna().unique())
        fields.append(f)
    return fields


def _report(spec, handoff: ValidatorHandoff, df: pd.DataFrame, build: Path, from_mlflow: bool,
            classes: list) -> dict:
    """Le rapport d'analyse v1.0 de l'agent (agent_api_docker/schemas.py), construit depuis le contrat."""
    features = [c for c in (spec.feature_columns or df.columns) if c != spec.target]
    binary = spec.task == "classification" and len(classes) == 2
    versions = _versions() if from_mlflow else {}
    packages = [{"name": n, "version": v} for n, v in versions.items()]
    packages += [{"name": n, "version": v} for n, v in spec.pinned_dependencies.items()
                 if n.lower() not in versions]
    py = f"{sys.version_info.major}.{sys.version_info.minor}" if from_mlflow else (spec.python_version or "3.12")
    py = ".".join(py.split(".")[:2])

    if from_mlflow:
        ml = {"framework": "sklearn", "framework_version": versions["scikit-learn"], "model_file": MODEL_FILE,
              "model_exists_in_project": True, "model_format": "joblib", "loader": "joblib.load"}
        entry = {"file": PREDICT_MODULE, "function": "predict"}
    else:  # repli : le modèle du projet, entraîné au build comme dans le pipeline d'origine de l'agent
        train = spec.entrypoints.get("train", "train.py").removesuffix(".py")
        ml = {"framework": "sklearn", "model_file": spec.model_artifact or "model.joblib",
              "model_exists_in_project": (build / (spec.model_artifact or "model.joblib")).exists(),
              "model_generation": {"required_before_serving": True, "recommended_stage": "docker_build",
                                   "command": f"python -c \"from {train} import train; train('{spec.data_path}', '.')\"",
                                   "requires_files": [spec.data_path]},
              "model_format": "joblib", "loader": "joblib.load"}
        loader = "load_model" if "load_model" in spec.interfaces else None
        entry = {"file": spec.entrypoints.get("predict", "predict.py"), "function": "predict",
                 "loader_function": loader}

    example = json.loads(df[features].head(1).to_json(orient="records"))[0]
    return {
        "schema_version": "1.0",
        "project_name": spec.project_name,
        "needs_user_input": False,
        "python": {"version": py, "package_manager": "pip"},
        "dependencies": {"file": "requirements.txt", "status": "reliable", "packages": packages},
        "ml": ml,
        "inference": {"entrypoint": entry, "supports_proba": binary, "preprocessing_location": "inside_model"},
        "io": {
            "input_type": "tabular",
            "input_schema": _input_schema(df, features),
            "output_schema": [{"name": "prediction", "type": "int"}]
                             + ([{"name": "proba", "type": "float", "optional": True}] if binary else []),
            "class_labels": classes,
            "example_input": example,
            "schema_confidence": "high",
        },
        "resources": {"gpu_required": "no"},
        "existing_app": {"has_api": False, "port": CONTAINER_PORT},
        "project": {
            "file_tree": sorted(str(p.relative_to(build)) for p in build.rglob("*") if p.is_file()),
            "ignorable_paths": IGNORE + [spec.data_path],
            "runtime_files": [PREDICT_MODULE, MODEL_FILE] if from_mlflow else [entry["file"], ml["model_file"]],
        },
        "warnings": [c.message for c in spec.carry_forward][:10],
        "unresolved": [],
    }


def _example_output(build: Path, report: dict, binary: bool) -> dict:
    """Sortie attendue pour l'exemple, calculée avec le modèle lui-même (sert au test test_predict_example)."""
    import importlib.util

    import joblib

    spec = importlib.util.spec_from_file_location("mlflow_predict", build / PREDICT_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = joblib.load(build / MODEL_FILE)
    x = report["io"]["example_input"]
    out = {"prediction": int(mod.predict(model, x)[0])}
    if binary:
        out["proba"] = float(model.predict_proba(pd.DataFrame([x]))[0, 1])
    return out


def _serve(image: str, slug: str, port: int, warnings: list[str]) -> str | None:
    """Démarre le conteneur de l'image validée (remplace celui d'un déploiement précédent) ; renvoie l'endpoint."""
    from .tools.checks import healthcheck
    from .tools.docker import DockerCLI

    d, name = DockerCLI(), f"mlops-{slug}"
    d.cleanup(name)
    run = d.run(image, CONTAINER_PORT, name=name, host_port=port)
    if run["exit_code"] != 0 or not run.get("host_port"):
        warnings.append(f"conteneur non démarré sur le port {port} : {run['logs'][-300:]}")
        return None
    base = f"http://127.0.0.1:{run['host_port']}"
    hc = healthcheck(base + "/health", 60, alive=lambda: d.state(run["container_id"])["status"] == "running")
    if not hc["ok"]:
        warnings.append(f"/health ne répond pas : {hc['detail']}")
        return None
    return base


def _promote(meta, warnings: list[str]) -> None:
    try:
        from mlflow_agent.orchestrator_adapter import promote

        promote(meta.model_name, meta.model_version)
        warnings.append(f"MLflow : {meta.model_name} v{meta.model_version} est maintenant le champion")
    except Exception as exc:  # noqa: BLE001 - le déploiement reste valable, l'alias est une information
        warnings.append(f"alias champion non posé ({type(exc).__name__}: {str(exc)[:150]})")


def _interactive() -> bool:
    """Un humain répond dans ce terminal (pas --auto-approve, ni Streamlit, ni les tests)."""
    return os.environ.get("ORCH_INTERACTIVE") == "1" and sys.stdin.isatty()


def _credentials_prompt(project_name: str, input_fn=input, secret_fn=getpass.getpass, out=print):
    """Choix du compte Docker Hub avant le push, comme le CLI de l'agent (terminal.py).

    Entrée = compte de agents/agent3/.env (son token) ; un autre nom = ce compte, token demandé sans écho.
    Rien n'est écrit sur disque, ni dans la base de l'orchestrateur, ni dans les traces.
    """
    from . import config
    from .docker_stage import image_slug
    from .terminal import make_credentials_prompt

    default = config.DOCKERHUB_USERNAME
    retry = make_credentials_prompt(default, input_fn=input_fn, secret_fn=secret_fn, out=out)

    def ask(reason: str | None = None):
        if reason is not None:                     # connexion refusée : on redemande le token (logique de l'agent)
            return retry(reason)
        ns = config.DOCKER_IMAGE_NAMESPACE_EXPLICIT or default or "<compte>"
        out("\n── Agent 3 : publication sur Docker Hub ──")
        out(f"Destination par défaut : docker.io/{ns}/{image_slug(project_name)}  (dépôt PUBLIC s'il n'existe pas)")
        try:
            label = f"Compte Docker Hub [{default}] (Entrée = compte du .env) : " if default else "Compte Docker Hub : "
            user = input_fn(label).strip() or default
            if not user:
                out("Aucun compte : push annulé.")
                return None
            if user == default and config.DOCKERHUB_TOKEN:
                return user, config.DOCKERHUB_TOKEN
            token = secret_fn(f"Token Docker Hub de {user} (droits Read & Write, non affiché) : ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return None
        if not token:
            out("Aucun token : push annulé.")
            return None
        return user, token

    return ask


def _confirm_push(req, input_fn=input, out=print) -> bool:
    out(f"Image testée : {req.image_id[:19]}...  ({req.size_mb} Mo)")
    out(f"Destination  : docker.io/{req.repository}  (tags : {', '.join(req.tags)})")
    try:
        return input_fn("Pousser cette image sur Docker Hub ? [o/N] ").strip().lower() in ("o", "oui", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _publish_choice(inp: AgentInput) -> bool | None:
    """Choix fait dans Slack à la gate champion (runs/<run_id>/publish.json) : publier ou non. None : pas de choix."""
    f = Path(inp.workdir).parent / "publish.json"
    try:
        return bool(json.loads(f.read_text(encoding="utf-8")).get("push")) if f.is_file() else None
    except (OSError, ValueError):
        return None


def _push_wanted(warnings: list[str]) -> bool:
    """AGENT3_PUSH=1 (environnement ou agents/agent3/.env) et identifiants Docker Hub disponibles."""
    from . import config
    from .docker_stage import missing_push_settings

    if config._get("AGENT3_PUSH").lower() not in {"1", "true", "yes"}:
        return False
    if _interactive():                             # les identifiants seront demandés dans le terminal
        return True
    missing = missing_push_settings()
    if missing:
        warnings.append("publication Docker Hub demandée mais réglages absents dans agents/agent3/.env : "
                        + ", ".join(missing))
        return False
    return True


def _checks(result) -> list[CheckResult]:
    checks = []
    if result.docker is not None:
        d, stage = result.docker, result.docker.stage
        order = ["lint", "build", "run", "health", "tests", "done"]
        reached = order.index(stage) if stage in order else -1
        for name, at in (("hadolint", "lint"), ("build", "build"), ("startup", "run"), ("health", "health"),
                         ("inference", "tests")):
            ok = d.ok or reached > order.index(at)
            checks.append(CheckResult(name=name, passed=ok, details=d.failure[:300] if not ok else ""))
    else:  # API testée en local seulement
        t = result.tests
        checks.append(CheckResult(name="startup", passed=bool(t and t.started), details="uvicorn local"))
        checks.append(CheckResult(name="inference", passed=result.api_ok,
                                  details=f"{t.passed} tests passés, {t.failed} échecs" if t else "tests non lancés"))
    return checks


def run(inp: AgentInput) -> AgentResult:
    from .loop import run_pipeline
    from .tools.docker import DockerCLI

    started = time.monotonic()
    workdir = Path(inp.workdir)
    warnings: list[str] = []
    spec = read_output(inp, "project_spec")
    handoff = ValidatorHandoff(**json.loads(Path(inp.report_path).read_text(encoding="utf-8"))["handoff"])
    if spec.task != "classification":
        return AgentResult(agent="agent3", status=AgentStatus.FAILED, can_proceed=False,
                           reasons=[("le modèle d'API de l'agent ne gère que la classification (prediction: int) : "
                                     "régression à ajouter au template api.py.j2")],
                           duration_s=round(time.monotonic() - started, 3))

    build = _writable_copy(Path(inp.project_dir), workdir / "project")
    df = pd.read_csv(build / spec.data_path)
    classes = sorted(df[spec.target].dropna().unique().tolist())
    try:
        classes = [int(c) for c in classes]
    except (TypeError, ValueError):
        return AgentResult(agent="agent3", status=AgentStatus.FAILED, can_proceed=False,
                           reasons=[f"classes non entières ({classes[:5]}) : l'API renvoie prediction: int"],
                           duration_s=round(time.monotonic() - started, 3))

    meta = read_output(inp, "mlflow_metadata") if "mlflow_metadata" in inp.inputs else None
    from_mlflow = False
    if meta and meta.model_uri:
        try:
            _export_mlflow_model(meta, build)
            mapping = classes if classes != list(range(len(classes))) else []
            (build / PREDICT_MODULE).write_text(_PREDICT_SRC.format(classes=mapping), encoding="utf-8")
            from_mlflow = True
        except Exception as exc:  # noqa: BLE001 - registre injoignable, modèle illisible : repli sur le projet
            warnings.append(f"modèle MLflow {meta.model_uri} non chargé ({type(exc).__name__}: {str(exc)[:200]}) : "
                            "modèle du projet utilisé")
    else:
        warnings.append("pas de mlflow_metadata : modèle du projet utilisé")

    report = _report(spec, handoff, df, build, from_mlflow, classes)
    binary = len(classes) == 2
    if from_mlflow:
        report["io"]["example_output"] = _example_output(build, report, binary)
    report_path = workdir / "analysis_report_v1.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    use_docker = os.environ.get("AGENT3_DOCKER", "1") != "0" and DockerCLI().available()[0]
    if not use_docker:
        warnings.append("Docker indisponible (démon arrêté ou AGENT3_DOCKER=0) : API testée en local, non déployée")
    interactive = _interactive()
    choice = None if interactive else _publish_choice(inp)      # terminal : la question est posée plus bas
    if choice is False:
        warnings.append("publication Docker Hub non demandée à la validation du champion")
    push = use_docker and choice is not False and _push_wanted(warnings)
    log = workdir / "agent3.log"
    with log.open("w", encoding="utf-8") as f:
        result = run_pipeline(
            spec.project_name, report_path=str(report_path), project_dir=build, workspace=workdir / "workspace",
            python_exe=sys.executable, agent="never", fix=os.environ.get("AGENT3_NO_FIX") != "1",
            docker=use_docker, push=push,
            approve_push=(_confirm_push if interactive else (lambda req: True)) if push else None,
            credentials=_credentials_prompt(spec.project_name) if push and interactive else None,
            on_step=lambda m: f.write(m + "\n"), on_detail=lambda m: f.write("  " + m + "\n"))

    generated = workdir / "workspace" / "generated"
    outputs = {"analysis_report_v1": str(report_path), "agent3_log": str(log),
               "agent3_result": str(workdir / "workspace" / "result.json")}
    outputs = {k: v for k, v in outputs.items() if Path(v).exists()}
    if not result.api_ok or (result.docker is not None and not result.docker.ok):
        t = result.tests
        why = (result.docker.failure if result.docker and not result.docker.ok else
               result.validation.summary() if not result.validation.ok else
               f"tests de l'API : {t.failed} échec(s)" if t else "tests non lancés")
        return AgentResult(agent="agent3", status=AgentStatus.FAILED, can_proceed=False, outputs=outputs,
                           reasons=[f"API ou image non validée : {why}"[:500]], warnings=warnings,
                           duration_s=round(time.monotonic() - started, 3))

    pushed = bool(result.push and result.push.pushed)
    if result.push is not None and not pushed:
        warnings.append(f"publication Docker Hub non faite : {result.push.error or 'refusée'}"[:300])
    version = (meta.model_version if meta and from_mlflow else "") or "projet"
    slug, endpoint = result.project_id.lower().replace("_", "-"), None
    image, tag = f"mlops/{slug}", f"v{version}-{inp.run_id}"
    if result.docker is not None and result.docker.ok:
        from .docker_stage import image_slug
        from .tools.docker import DockerCLI as _D

        slug = image_slug(spec.project_name)
        image = f"mlops/{slug}"
        _D().tag(result.docker.image, f"{image}:{tag}")
        endpoint = _serve(f"{image}:{tag}", slug, int(os.environ.get("AGENT3_PORT", "8000")), warnings)
        if endpoint and from_mlflow:
            _promote(meta, warnings)

    dep = DeploymentResult(
        model_uri=meta.model_uri if from_mlflow else f"project:{report['ml']['model_file']}",
        model_source="mlflow" if from_mlflow else "project_artifact",
        image=result.push.repository if pushed else image,
        tag=result.push.tags[0] if pushed else tag,
        checks=_checks(result), pushed=pushed, endpoint=endpoint,
        generated_files={n: str(generated / n) for n in ("api.py", "Dockerfile", "requirements.api.txt",
                                                          ".dockerignore", PREDICT_MODULE)
                         if (generated / n).exists()},
    )
    outputs["deployment_result"] = write_output(inp, "deployment_result", dep)
    if endpoint is None and use_docker:
        warnings.append("image validée mais conteneur non démarré : endpoint vide")
    return AgentResult(agent="agent3", status=AgentStatus.OK if endpoint else AgentStatus.WARNING,
                       can_proceed=True, outputs=outputs, warnings=warnings,
                       duration_s=round(time.monotonic() - started, 3))
