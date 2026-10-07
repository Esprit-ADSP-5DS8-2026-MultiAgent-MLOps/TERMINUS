"""Exécuté DANS le sandbox. Appelle train/load_model/predict du projet selon le contrat et écrit un rapport JSON.
N'utilise que la bibliothèque standard + pandas (déjà requis par le contrat)."""
import argparse
import importlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project"), ap.add_argument("--manifest"), ap.add_argument("--out1"), ap.add_argument("--out2")
    ap.add_argument("--report"), ap.add_argument("--no-determinism", action="store_true")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    project = os.path.abspath(a.project)
    os.chdir(project)
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
        train_mod = step("import_train", lambda: importlib.import_module("train"))
        pred_mod = step("import_predict", lambda: importlib.import_module("predict"))
        df = step("load_data", lambda: pd.read_csv(data_path))
        X = df.drop(columns=[man["target"]]) if df is not None and man["target"] in df.columns else None

        metrics = step("train", lambda: train_mod.train(data_path, a.out1)) if train_mod else None
        rep["metrics"] = metrics if isinstance(metrics, dict) else None
        rep["train_return_type"] = type(metrics).__name__
        artifact = Path(a.out1) / "model.joblib"
        rep["artifact_exists"] = artifact.exists()

        model = step("load_model", lambda: pred_mod.load_model(str(artifact))) if (pred_mod and artifact.exists()) else None
        if model is not None and X is not None:
            res = step("predict_df", lambda: pred_mod.predict(model, X.head(5)))
            if res is not None:
                rep["predict_len"] = len(res) if hasattr(res, "__len__") else None
                rep["predict_type"] = type(res).__name__
                try:
                    json.dumps(res)
                    rep["predict_json_ok"] = True
                except TypeError:
                    rep["predict_json_ok"] = False
            res1 = step("predict_dict", lambda: pred_mod.predict(model, X.iloc[0].to_dict()))
            rep["predict_dict_len"] = len(res1) if res1 is not None and hasattr(res1, "__len__") else None

        if train_mod and metrics is not None and not a.no_determinism:
            m2 = step("train_2", lambda: train_mod.train(data_path, a.out2))
            rep["metrics2"] = m2 if isinstance(m2, dict) else None
    except BaseException:
        rep["fatal"] = traceback.format_exc()[-800:]
    rep["peak_mem_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    json.dump(rep, open(a.report, "w"), default=str)


if __name__ == "__main__":
    main()
