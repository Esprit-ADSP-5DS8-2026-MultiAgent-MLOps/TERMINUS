"""Usage:
  python -m mlflow_agent report.json                         (plan only)
  python -m mlflow_agent report.json --base-dir DIR --execute [--no-llm] [--out result.json]
The LLM comes from --model, else $MLFLOW_AGENT_LLM, else mlflow_agent.toml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlflow_agent.config import ConfigError, resolve_llm
from mlflow_agent.llm_client import make_client
from mlflow_agent.planner import default_plan, plan_experiment
from mlflow_agent.schemas import (
    InvalidReportError, NotReadyError, load_report, require_ready, resolve_paths,
)


def main() -> None:
    p = argparse.ArgumentParser(prog="mlflow_agent")
    p.add_argument("report", help="path to Agent 1's JSON report")
    p.add_argument("--model", default=None,
                   help="qwen2.5:3b (Ollama) or gemini:<model>. Default: mlflow_agent.toml")
    p.add_argument("--no-llm", action="store_true", help="use the default plan, no LLM call")
    p.add_argument("--execute", action="store_true", help="train, tune, track and gate")
    p.add_argument("--base-dir", default=".", help="directory the report's paths refer to")
    p.add_argument("--out", default="result.json")
    p.add_argument("--experiment", default="mlflow_agent")
    args = p.parse_args()

    try:
        report = load_report(args.report)
        require_ready(report)
        paths = resolve_paths(report, args.base_dir) if args.execute else None
    except (InvalidReportError, NotReadyError) as e:
        print(f"REJECTED: {e}")
        sys.exit(1)

    notes: list[str] = []
    if args.no_llm:
        plan, source, llm = default_plan(report), "fallback", "none"
        notes.append("default plan used (--no-llm)")
    else:
        try:
            llm, where = resolve_llm(args.model)
            client = make_client(llm)
        except (ConfigError, ValueError) as e:
            print(f"config error: {e}")
            sys.exit(1)
        except Exception as e:  # e.g. missing API key
            print(f"LLM setup error: {e}")
            sys.exit(1)
        print(f"LLM: {llm}  (from {where})")
        pr = plan_experiment(report, client)
        plan, source = pr.plan, pr.source
        notes += [f"LLM error: {e}" for e in pr.errors]
        if source == "fallback":
            notes.append("LLM plan unavailable or invalid: fallback plan used")
    print(f"plan source: {source}")
    for note in notes:
        print(f"  note: {note}")
    print(plan.model_dump_json(indent=2))

    if args.execute:
        from mlflow_agent.executor import run_experiment  # heavy imports only when needed
        result = run_experiment(report, plan, paths, experiment_name=args.experiment,
                                warnings_in=notes)
        Path(args.out).write_text(result.model_dump_json(indent=2))
        print(f"\nstatus: {result.status}   (saved: {args.out})")
        print(f"best: {result.best_model_name}  val={result.validation_score}  "
              f"test={result.test_score}  baseline={result.baseline_test_score}")
        print(f"model uri: {result.model_uri}")
        for w in result.warnings:
            print(f"  warning: {w}")
        for e in result.errors:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
