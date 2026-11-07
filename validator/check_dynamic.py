"""Niveau 5 - Dynamique : exécute le projet dans un sandbox sur une COPIE temporaire.
Ne s'exécute que si les niveaux 1 à 4 n'ont aucun bloquant (le code n'est jamais lancé avant d'être jugé sûr)."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from importlib import metadata
from pathlib import Path

from .contract import ARTIFACT_EXTENSIONS, LOWER_IS_BETTER, MODEL_ARTIFACT
from .fs import IGNORED_DIRS
from .models import Severity as S
from .sandbox import ensure_venv, pick_sandbox

HARNESS = Path(__file__).with_name("harness.py")


def _ignore(_dir, names):
    return [n for n in names if n in IGNORED_DIRS or Path(n).suffix.lower() in ARTIFACT_EXTENSIONS]


def _snapshot(root: Path):
    snap = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            p = Path(dp) / f
            st = p.stat()
            snap[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return snap


def _tail(text: str, n: int = 400) -> str:
    return (text or "").strip()[-n:]


def run(ctx):
    cfg, opts = ctx.cfg, ctx.options
    sb, note = pick_sandbox(opts.get("sandbox", "auto"), ctx.root, cfg.memory_mb)
    ctx.sandbox_info = sb.describe()
    if note:
        ctx.add("C12", "C12.SANDBOX_FALLBACK", S.INFO, note)
    if not sb.network_isolated:
        ctx.add("C11", "C11.NETWORK_NOT_ISOLATED", S.INFO, "Réseau non isolé dans ce sandbox (unshare indisponible) : utilisez Docker en production.")
    env_mode = opts.get("env", "venv")
    if sb.name == "subprocess" and env_mode == "venv":
        # venv isolé construit depuis le requirements.txt du PROJET : plus aucune dépendance à installer à la main
        py, err = ensure_venv(ctx.root / "requirements.txt", opts.get("cache_dir"))
        if err:
            ctx.add("C6", "C6.INSTALL_FAILED", S.BLOCKER, "Impossible d'installer les dépendances épinglées dans un environnement isolé.",
                    file="requirements.txt", evidence=err,
                    hint="Vérifiez que chaque version existe en wheel pour votre version de Python et que le réseau est disponible.")
            return
        sb.python = str(py)
        ctx.sandbox_info = sb.describe()
        ctx.facts["env"] = {"mode": "venv", "python": str(py)}
    elif sb.name == "subprocess":  # mode "current" : l'environnement local doit refléter les versions épinglées
        ctx.facts["env"] = {"mode": "current"}
        bad = []
        for name, ver in (ctx.facts.get("requirements") or {}).items():
            try:
                got = metadata.version(name)
                if ver and got != ver:
                    bad.append(f"{name} {got} (épinglé {ver})")
            except metadata.PackageNotFoundError:
                bad.append(f"{name} non installé")
        if bad:
            ctx.add("C6", "C6.ENV_MISMATCH", S.WARNING, "Environnement d'exécution différent des versions épinglées : résultats peu représentatifs.",
                    evidence="; ".join(bad[:5]), hint="Utilisez --env venv (défaut) ou le backend Docker.")

    work = Path(tempfile.mkdtemp(prefix="cv_"))
    try:
        _execute(ctx, sb, work)
    finally:
        if not opts.get("keep_workdir"):
            shutil.rmtree(work, ignore_errors=True)


def _execute(ctx, sb, work: Path):
    cfg, opts, m = ctx.cfg, ctx.options, ctx.manifest
    timeout = int(opts.get("timeout", cfg.dynamic_timeout_s))
    proj = work / "project"
    shutil.copytree(ctx.root, proj, symlinks=True, ignore=_ignore)
    (work / "harness").mkdir()
    shutil.copy(HARNESS, work / "harness" / "_harness.py")
    for d in ("out_main", "out_h1", "out_h2", "tmp"):
        (work / d).mkdir()
    (work / "manifest.json").write_text(json.dumps({
        "data_path": m.data.path, "target": m.data.target, "task": m.project.task, "primary_metric": m.evaluation.primary_metric}))
    before = _snapshot(proj)
    dyn = ctx.facts.setdefault("dynamic", {})

    # --- 1) main.py (C5, C12) ----------------------------------------------------------------------
    r = sb.run(["python", "main.py"], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, timeout)
    dyn["main_s"] = round(r.duration_s, 2)
    if r.timed_out:
        ctx.add("C12", "C12.TIMEOUT", S.BLOCKER, f"main.py dépasse {timeout}s.", file="main.py", hint="Réduisez les données/hyperparamètres ou le coût d'entraînement.")
    elif r.returncode != 0:
        ctx.add("C5", "C5.MAIN_FAILED", S.BLOCKER, f"main.py a échoué (code {r.returncode}).", file="main.py", evidence=_tail(r.stderr),
                hint="Corrigez l'erreur ci-dessus ; main.py doit s'exécuter sans argument ni interaction.")
    elif not (work / "out_main" / MODEL_ARTIFACT).exists():
        ctx.add("C5", "C5.MAIN_NO_ARTIFACT", S.WARNING, f"main.py n'a pas écrit {MODEL_ARTIFACT} dans $OUTPUT_DIR.", file="main.py",
                hint="Écrivez les sorties dans os.environ['OUTPUT_DIR'].")

    # --- 2) harnais : train / load_model / predict (C3, C4, C9) -----------------------------------------
    args = ["python", "{W}/harness/_harness.py", "--project", "{W}/project", "--manifest", "{W}/manifest.json",
            "--out1", "{W}/out_h1", "--out2", "{W}/out_h2", "--report", "{W}/report.json"]
    if not opts.get("determinism", True):
        args.append("--no-determinism")
    h = sb.run(args, work, proj, {}, timeout)
    dyn["harness_s"] = round(h.duration_s, 2)
    rep_path = work / "report.json"
    if h.timed_out:
        ctx.add("C12", "C12.TIMEOUT", S.BLOCKER, f"train/predict dépasse {timeout}s.", hint="Réduisez le coût d'entraînement.")
    elif not rep_path.exists():
        ctx.add("C3", "C3.HARNESS_FAILED", S.BLOCKER, "Le harnais n'a produit aucun rapport (crash du sandbox).", evidence=_tail(h.stderr))
    else:
        _interpret(ctx, json.loads(rep_path.read_text()), work)

    # --- 3) écritures hors OUTPUT_DIR (C10) ----------------------------------------------------------------------
    after = _snapshot(proj)
    changed = sorted(k for k, v in after.items() if before.get(k) != v)
    if changed:
        ctx.add("C10", "C10.WRITES_OUTSIDE_OUTPUT_DIR", S.BLOCKER, "Le projet écrit dans son propre dossier au lieu de output_dir.",
                evidence=", ".join(changed[:5]), hint="Toutes les sorties doivent aller dans output_dir / $OUTPUT_DIR.")


def _interpret(ctx, rep, work):
    m, steps = ctx.manifest, rep.get("steps", {})
    dyn = ctx.facts.setdefault("dynamic", {})
    dyn["peak_mem_mb"] = rep.get("peak_mem_mb")
    dyn["train_s"] = (steps.get("train") or {}).get("s")

    def failed(name, rule, code, msg, hint=None):
        s = steps.get(name)
        if s is not None and not s["ok"]:
            ctx.add(rule, code, S.BLOCKER, f"{msg} : {s['error']}", evidence=s.get("where"), hint=hint)
            return True
        return False

    if rep.get("fatal"):
        ctx.add("C3", "C3.HARNESS_FAILED", S.BLOCKER, "Erreur interne du harnais.", evidence=rep["fatal"][-300:])
    failed("import_train", "C3", "C3.IMPORT_FAILED", "train.py n'est pas importable", "Vérifiez les imports et les dépendances.")
    failed("import_predict", "C4", "C4.IMPORT_FAILED", "predict.py n'est pas importable")
    if failed("train", "C3", "C3.TRAIN_FAILED", "train() a échoué", "Corrigez l'erreur ; train(data_path, output_dir) doit s'exécuter sur data.csv."):
        return

    metrics = rep.get("metrics")
    if "train" in steps and steps["train"]["ok"]:
        if metrics is None:
            ctx.add("C3", "C3.BAD_RETURN", S.BLOCKER, f"train() doit retourner un dict de métriques (reçu : {rep.get('train_return_type')}).")
        else:
            val = metrics.get(m.evaluation.primary_metric)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                ctx.add("C3", "C3.MISSING_PRIMARY_METRIC", S.BLOCKER,
                        f"La métrique `{m.evaluation.primary_metric}` du manifest est absente/invalide dans le dict retourné.",
                        evidence=f"clés reçues : {sorted(metrics)}")
            else:
                ctx.observed_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                # rmse / mae : plus bas = meilleur, le seuil est une erreur MAXIMALE (même règle que le quality gate)
                lower_better = m.evaluation.primary_metric in LOWER_IS_BETTER
                if (val > m.evaluation.min_threshold) if lower_better else (val < m.evaluation.min_threshold):
                    ctx.add("C8", "C8.METRIC_BELOW_THRESHOLD", S.WARNING,
                            f"{m.evaluation.primary_metric}={val:.4f} {'>' if lower_better else '<'} seuil "
                            f"{m.evaluation.min_threshold} : le quality gate rejetterait ce modèle.")
        if not rep.get("artifact_exists"):
            ctx.add("C3", "C3.NO_MODEL_ARTIFACT", S.BLOCKER, f"train() n'a pas écrit {MODEL_ARTIFACT} dans output_dir.", hint=f"joblib.dump(model, output_dir/'{MODEL_ARTIFACT}')")

    failed("load_model", "C4", "C4.LOAD_FAILED", "load_model() a échoué", "Les classes personnalisées doivent être importables (importez-les dans predict.py).")
    failed("predict_df", "C4", "C4.PREDICT_FAILED", "predict(model, DataFrame) a échoué")
    failed("predict_dict", "C4", "C4.PREDICT_DICT_FAILED", "predict(model, dict) a échoué", "predict doit accepter un dict (une ligne) ET un DataFrame.")
    if steps.get("predict_df", {}).get("ok"):
        if rep.get("predict_len") != 5:
            ctx.add("C4", "C4.BAD_OUTPUT", S.BLOCKER, f"predict() sur 5 lignes doit retourner 5 prédictions (reçu : {rep.get('predict_len')}, type {rep.get('predict_type')}).")
        if rep.get("predict_json_ok") is False:
            ctx.add("C4", "C4.NOT_SERIALIZABLE", S.BLOCKER, "Sortie de predict() non sérialisable en JSON.", hint="Retournez `.tolist()`.")
    if steps.get("predict_dict", {}).get("ok") and rep.get("predict_dict_len") != 1:
        ctx.add("C4", "C4.BAD_OUTPUT_DICT", S.BLOCKER, f"predict(model, dict) doit retourner 1 prédiction (reçu : {rep.get('predict_dict_len')}).")

    m2 = rep.get("metrics2")
    if rep.get("metrics") and m2:
        diffs = {k: abs(rep["metrics"][k] - m2[k]) for k in rep["metrics"]
                 if isinstance(rep["metrics"][k], (int, float)) and isinstance(m2.get(k), (int, float)) and abs(rep["metrics"][k] - m2[k]) > 1e-9}
        if diffs:
            ctx.add("C9", "C9.NON_DETERMINISTIC", S.WARNING, "Deux entraînements successifs donnent des métriques différentes.",
                    evidence=", ".join(f"{k}: Δ={v:.4f}" for k, v in list(diffs.items())[:4]), hint="Fixez toutes les graines (split, modèles, numpy).")
        else:
            dyn["deterministic"] = True
