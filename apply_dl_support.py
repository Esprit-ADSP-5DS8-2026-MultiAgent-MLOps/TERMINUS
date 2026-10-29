#!/usr/bin/env python3
"""Ajoute : (1) le profil `deep-learning` (torch/tensorflow/keras dans le périmètre),
(2) l'exécution depuis pyproject.toml / environment.yml quand requirements.txt est absent,
(3) une limite mémoire du sandbox configurable par contrat. `validator/` (baseline) n'est pas touché. Idempotent."""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent


def patch(rel, old, new, marker):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print(f"= {rel} (déjà appliqué)")
        return
    if old not in s:
        sys.exit(f"!! {rel} : ancre introuvable, fichier différent de l'archive analysée")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"+ {rel}")


# 1) contrat deep-learning : donnée pure, dérivée de training_only.yaml
tpl = (ROOT / "policy/contracts/training_only.yaml").read_text(encoding="utf-8")
dl = (tpl.replace("id: training-only", "id: deep-learning")
         .replace('version: "1.0"', 'version: "1.0"\nsandbox_memory_mb: 16384')
         .replace("frameworks_in_scope: [scikit-learn, pandas, numpy, scipy, joblib, xgboost, lightgbm]",
                  "frameworks_in_scope: [scikit-learn, pandas, numpy, scipy, joblib, xgboost, lightgbm, torch, torchvision, torchaudio, tensorflow, keras]")
         .replace("frameworks_out_of_scope: [torch, tensorflow, keras, jax, transformers]",
                  "frameworks_out_of_scope: [jax, mxnet]")
         .replace("# Profil « entraînement seul »", "# Profil « deep learning » (train() seul, torch/tensorflow/keras autorisés) — dérivé de training_only\n# Profil « entraînement seul »", 1))
(ROOT / "policy/contracts/deep_learning.yaml").write_text(dl, encoding="utf-8")
print("+ policy/contracts/deep_learning.yaml")

# 2) champ de contrat : mémoire du sandbox
patch("policy/model.py", "    execution_required: bool = True\n",
      "    execution_required: bool = True\n    sandbox_memory_mb: Optional[int] = None          # torch/tensorflow : la limite par défaut (4 Go d'espace d'adressage) est trop basse\n",
      "sandbox_memory_mb")

# 3) le Validator transmet cette limite au sandbox
patch("validator_agent/agent.py",
      "cache_dir=cfg.cache_dir, timeout=cfg.timeout))\n    py_out",
      "cache_dir=cfg.cache_dir, timeout=cfg.timeout,\n                                                                memory_mb=contract.sandbox_memory_mb or 4096))\n    py_out",
      "contract.sandbox_memory_mb")

# 4) sandbox : dépendances dérivées de pyproject.toml / environment.yml
HELPER = r'''

def derive_requirements(root: Path, cache_dir=None) -> Path | None:
    """requirements.txt absent : dérive les dépendances de pyproject.toml ([project].dependencies) ou d'environment.yml
    (section pip + paquets conda `nom=ver` -> `nom==ver`). Fichier hors du projet, jamais écrit dedans.
    Les paquets non épinglés restent tels quels : c'est ENV.DEPENDENCIES_PINNED qui les signale."""
    lines: list[str] = []
    py = root / "pyproject.toml"
    if py.is_file():
        try:
            import tomllib
            lines = [str(d) for d in (tomllib.loads(py.read_text(encoding="utf-8")).get("project", {}).get("dependencies") or [])]
        except Exception:
            lines = []
    if not lines:
        for n in ("environment.yml", "environment.yaml"):
            f = root / n
            if not f.is_file():
                continue
            try:
                import yaml
                for d in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("dependencies") or []:
                    if isinstance(d, dict):
                        lines += [str(x) for x in d.get("pip") or []]
                    elif isinstance(d, str):
                        d = d.split("::")[-1].strip()
                        if re.split(r"[=<>! ]", d)[0].lower() in ("python", "pip", ""):
                            continue
                        lines.append(re.sub(r"^([A-Za-z0-9_.\-]+)=([^=].*)$", r"\1==\2", d))
            except Exception:
                lines = []
            break
    lines = [x for x in lines if x.strip()]
    if not lines:
        return None
    text = "\n".join(lines) + "\n"
    out = Path(cache_dir or os.environ.get("CV_CACHE_DIR") or Path.home() / ".cache" / "contract-validator") / "derived"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"req_{hashlib.sha256(text.encode()).hexdigest()[:16]}.txt"
    f.write_text(text, encoding="utf-8")
    return f
'''
patch("agentic_core/sandbox_exec.py", "import json\nimport math", "import hashlib\nimport json\nimport math", "import hashlib")
patch("agentic_core/sandbox_exec.py", "\n\ndef num(d):", HELPER + "\n\ndef num(d):", "def derive_requirements")
patch("agentic_core/sandbox_exec.py",
      '        key = (str(root), req.read_bytes() if req.is_file() else None)',
      '        dep = req if req.is_file() else derive_requirements(root, self.cache_dir)\n        key = (str(root), dep.read_bytes() if dep else None)',
      "dep = req if")
patch("agentic_core/sandbox_exec.py",
      '''                if not req.is_file():
                    return ("requirements.txt absent : impossible de construire l'environnement d'exécution isolé "
                            "(V1 : seuls les projets installables depuis requirements.txt sont exécutés)")
                py, err = ensure_venv(req, self.cache_dir)''',
      '''                if dep is None:
                    return ("requirements.txt absent : impossible de construire l'environnement d'exécution isolé "
                            "(aucune dépendance exploitable dans requirements.txt, pyproject.toml ou environment.yml)")
                py, err = ensure_venv(dep, self.cache_dir, timeout=3600 if self.memory_mb > 4096 else 900)''',
      "if dep is None")
print("OK")
