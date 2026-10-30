"""Show everything the agent produced: input, plan, Optuna trials, result, benchmark."""
import json
import subprocess
import sys

import mlflow

REPORT = "examples/credit_risk_report.json"
RESULT = "result.json"


def title(text: str) -> None:
    print("\n" + "=" * 10 + f" {text} " + "=" * 10)


# 1. Input
title("1. INPUT: Agent 1's report (" + REPORT + ")")
rep = json.load(open(REPORT))
print("task:", rep["task"]["type"], "| target:", rep["task"]["target_column"],
      "| metric:", rep["task"]["primary_metric"], "| threshold:", rep["task"].get("min_primary_metric"))
print("rows:", rep["dataset_analysis"]["rows"], "| classes:", rep["dataset_analysis"]["class_distribution"])
print("budget:", rep["constraints"])

res = json.load(open(RESULT))

# 2. LLM plan
title("2. THE LLM'S PLAN (validated by code)")
print("models:", res["plan"]["models"])
print("max_trials:", res["plan"]["max_trials"])
print("rationale:", res["plan"]["rationale"])
print("warnings:", res["warnings"])

# 3. Optuna trials, read back from MLflow
title("3. OPTUNA TRIALS (every try, from MLflow)")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_ids=[res["experiment_id"]], output_format="pandas")
runs = runs[runs["tags.mlflow.parentRunId"].notna()].sort_values("start_time")
metric_cols = [c for c in runs.columns if c.startswith("metrics.val_")]
param_cols = [c for c in runs.columns if c.startswith("params.")]
show = ["tags.mlflow.runName", *metric_cols, *param_cols]
print(runs[show].rename(columns=lambda c: c.split(".", 1)[-1]).round(3).to_string(index=False))

# 4. Final result
title("4. FINAL RESULT (result.json, what Agent 3 reads)")
print("status:", res["status"])
print("best model:", res["best_model_name"], res["best_params"])
for k in ("train_score", "validation_score", "test_score", "baseline_test_score"):
    print(f"{k:22s}", round(res[k], 4))
print("quality gate:", res["quality_gate"]["checks"])
print("model uri:", res["model_uri"])

# 5. Benchmark
title("5. LLM + PROMPT BENCHMARK (36 cases each)")
subprocess.run([sys.executable, "-m", "mlflow_agent.compare"], check=False)
