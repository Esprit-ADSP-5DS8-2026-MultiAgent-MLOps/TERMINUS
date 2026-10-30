"""Write a clean summary of benchmark_results/ to docs/benchmark_report.md.

Usage: python -m mlflow_agent.report
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mlflow_agent.planner import _strip_fences
from mlflow_agent.plans import parse_plan


def reason(r: dict) -> str:
    if not r["raw"]:
        return "no answer from the model"
    try:
        parse_plan(_strip_fences(r["raw"]))
    except Exception as e:
        msg = str(e)
        if "search_method" in msg:
            return "search method not allowed"
        if "too_long" in msg:
            return "more than 3 models"
        if "models" in msg:
            return "model name not allowed"
        return "other invalid plan"
    return "wrong metric or budget"


def stats(d: dict) -> tuple:
    res = d["results"]
    n = len(res)
    ok = sum(r["all_passed"] for r in res)
    inj = [r for r in res if "injection" in r["tags"]]
    inj_ok = sum(r["all_passed"] for r in inj)
    sec = sum(r["latency_s"] for r in res) / n
    return ok, n, inj_ok, len(inj), sec


def main() -> None:
    runs = []
    for f in sorted(Path("benchmark_results").glob("*.json")):
        d = json.loads(f.read_text())
        if all(r["error"] for r in d["results"]):
            continue  # skip runs where the model never answered
        if len(d["results"]) < 36:
            continue  # only full 36-case runs are comparable
        runs.append(d)

    out = ["# Benchmark report", "",
           "Each run: the LLM writes one plan for each of 36 test projects. "
           "A plan passes only if it passes all six checks.", ""]

    out += ["## 1. Which LLM? (same prompt: v1-structured)", "",
            "| Model | Plans accepted | Injection tests passed | Seconds per plan |",
            "|---|---|---|---|"]
    for d in runs:
        if d["prompt_version"] == "v1-structured":
            ok, n, io, inn, sec = stats(d)
            out.append(f"| {d['model']} | {ok}/{n} ({ok/n:.0%}) | {io}/{inn} | {sec:.1f} |")

    out += ["", "## 2. Which prompt? (same model: qwen2.5:3b)", "",
            "| Prompt | Plans accepted | Injection tests passed |", "|---|---|---|"]
    for d in runs:
        if d["model"] == "qwen2.5:3b":
            ok, n, io, inn, _ = stats(d)
            out.append(f"| {d['prompt_version']} | {ok}/{n} ({ok/n:.0%}) | {io}/{inn} |")

    out += ["", "## 3. Why plans were rejected", ""]
    for d in runs:
        bad = [reason(r) for r in d["results"] if not r["all_passed"]]
        out.append(f"**{d['model']} / {d['prompt_version']}**: "
                   + ("none rejected" if not bad else
                      ", ".join(f"{k} ({v})" for k, v in Counter(bad).most_common())))
        out.append("")

    text = "\n".join(out)
    Path("docs/benchmark_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
