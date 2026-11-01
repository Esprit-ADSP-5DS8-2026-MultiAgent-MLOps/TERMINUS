"""Exécuté DANS le sandbox. Appelle train/load_model/predict du projet selon le contrat et écrit un rapport JSON.
N'utilise que la bibliothèque standard + pandas (déjà requis par le contrat)."""
import argparse
import importlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path


LEGACY = {
    "train": {"module": "train", "function": "train", "file": "train.py", "sys_path": "."},
    "load_model": {"module": "predict", "function": "load_model", "file": "predict.py", "sys_path": "."},
    "predict": {"module": "predict", "function": "predict", "file": "predict.py", "sys_path": "."},
}


def _entry(man, role):
    return (man.get("entrypoints") or {}).get(role) or LEGACY[role]


def _load_entry(project, e):
    """Importe le module de l'entrypoint (sys.path = dossier d'import résolu + racine du projet) et retourne la fonction."""
    for d in (e.get("sys_path") or "."), ".":
        full = os.path.normpath(os.path.join(project, d))
        if full not in sys.path:
            sys.path.insert(0, full)
    if e.get("module"):
        mod = importlib.import_module(e["module"])
    else:  # chemin non importable (tiret, etc.) : chargement par fichier
        spec = importlib.util.spec_from_file_location("_entry_" + os.path.basename(e["file"]).replace(".", "_"), os.path.join(project, e["file"]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return getattr(mod, e["function"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project"), ap.add_argument("--manifest"), ap.add_argument("--out1"), ap.add_argument("--out2")
    ap.add_argument("--report"), ap.add_argument("--no-determinism", action="store_true"), ap.add_argument("--out3")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    project = os.path.abspath(a.project)
    os.chdir(project)
    try:                                   # importé AVANT d'ajouter le projet à sys.path : un _audit.py candidat ne peut pas le masquer
        import _audit
        _audit.install(os.path.join(os.path.dirname(os.path.abspath(a.report)), "audit_harness.jsonl"), project)
    except ImportError:                    # harnais lancé hors Validator (ancien chemin) : pas de hook
        pass
    try:                                   # sonde comportementale de ML.METRICS_GENUINE (behavior_probe.py)
        import _probe
        _probe.arm()                       # AVANT l'import du candidat : ses `from sklearn.metrics import ...` passent par l'enregistreur
    except ImportError:
        _probe = None
    sys.path.insert(0, project)
    rep = {"steps": {}, "metrics": None, "metrics2": None, "predict_len": None, "peak_mem_mb": None}

    def step(name, fn):
        t0 = time.time()
        try:
            res = fn()
            rep["steps"][name] = {"ok": True, "s": round(time.time() - t0, 3)}
            return res
        except (Exception, SystemExit) as e:
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            rep["steps"][name] = {"ok": False, "s": round(time.time() - t0, 3), "error": tb[-500:],
                                  "where": traceback.format_exc().strip().splitlines()[-3][:200]}
            return None

    try:
        import pandas as pd
        data_path = str(Path(project) / man["data_path"])
        train_fn = step("import_train", lambda: _load_entry(project, _entry(man, "train")))

        def _load_pred():
            return _load_entry(project, _entry(man, "load_model")), _load_entry(project, _entry(man, "predict"))
        pred = step("import_predict", _load_pred)
        load_fn, predict_fn = pred if pred else (None, None)
        df = step("load_data", lambda: pd.read_csv(data_path))
        X = df.drop(columns=[man["target"]]) if df is not None and man["target"] in df.columns else None

        metrics = step("train", lambda: train_fn(data_path, a.out1)) if train_fn else None
        rep["metrics"] = metrics if isinstance(metrics, dict) else None
        rep["train_return_type"] = type(metrics).__name__
        artifact = Path(a.out1) / "model.joblib"
        rep["artifact_exists"] = artifact.exists()

        model = step("load_model", lambda: load_fn(str(artifact))) if (load_fn and artifact.exists()) else None
        if model is not None and X is not None:
            res = step("predict_df", lambda: predict_fn(model, X.head(5)))
            if res is not None:
                rep["predict_len"] = len(res) if hasattr(res, "__len__") else None
                rep["predict_type"] = type(res).__name__
                try:
                    json.dumps(res)
                    rep["predict_json_ok"] = True
                except TypeError:
                    rep["predict_json_ok"] = False
            res1 = step("predict_dict", lambda: predict_fn(model, X.iloc[0].to_dict()))
            rep["predict_dict_len"] = len(res1) if res1 is not None and hasattr(res1, "__len__") else None

        if train_fn and metrics is not None and not a.no_determinism:
            m2 = step("train_2", lambda: train_fn(data_path, a.out2))
            rep["metrics2"] = m2 if isinstance(m2, dict) else None
        if train_fn and metrics is not None and not a.no_determinism and a.out3 and _probe is not None \
                and rep["steps"].get("train", {}).get("s", 0) <= 60:        # budget : pas de 3e entraînement si le 1er est déjà long
            try:                           # entraînement de contrôle sur ÉTIQUETTES PERMUTÉES, avec enregistrement de ce que fait le code
                perm_path = _probe.permuted_copy(data_path, man["target"], os.path.join(a.out3, "perm_data.csv"))
                with _probe.Recorder() as rec:
                    pm = train_fn(perm_path, a.out3)
                rep["perm_metrics"] = pm if isinstance(pm, dict) else None
                rep["probe"] = rec.summary()
            except BaseException as e:     # la sonde ne doit jamais faire échouer le harnais : elle devient « non concluante »
                rep["probe_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    except BaseException:
        rep["fatal"] = traceback.format_exc()[-800:]
    rep["peak_mem_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    json.dump(rep, open(a.report, "w"), default=str)


if __name__ == "__main__":
    main()
