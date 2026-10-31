"""Stand-in for Agent 1: build an analysis report from a project's mlops.yaml + data.csv.
Usage: python examples/make_credit_risk_report.py <project_dir> [out.json]"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

proj = Path(sys.argv[1]).resolve()
out = sys.argv[2] if len(sys.argv) > 2 else "examples/credit_risk_report.json"
m = yaml.safe_load((proj / "mlops.yaml").read_text())
task, target = m["project"]["task"], m["data"]["target"]
df = pd.read_csv(proj / m["data"]["path"])
dist = ({str(k): int(v) for k, v in df[target].value_counts().items()}
        if task == "classification" else None)
na = df.isna().sum()
report = {
    "schema_version": "0.1",
    "project": {"project_path": ".", "dataset_path": m["data"]["path"], "training_script": None},
    "task": {"type": task, "target_column": target, "objective": m["project"].get("name", "project"),
             "primary_metric": m["evaluation"]["primary_metric"], "secondary_metrics": [],
             "min_primary_metric": m["evaluation"].get("min_threshold")},
    "dataset_analysis": {"rows": len(df), "columns": df.shape[1],
                         "missing_values": {"total": int(na.sum()),
                                            "columns_affected": [c for c in df.columns if na[c] > 0]},
                         "class_distribution": dist, "data_leakage_risks": []},
    "existing_implementation": {"training_code_available": True, "preprocessing_available": True,
                                "existing_models": []},
    "constraints": {"max_experiment_runs": 12, "max_runtime_minutes": 10},
    "status": "ready_for_experimentation",
}
Path(out).write_text(json.dumps(report, indent=2))
print("wrote", out)
