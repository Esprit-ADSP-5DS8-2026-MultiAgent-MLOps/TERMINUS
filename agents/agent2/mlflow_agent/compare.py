"""Clean comparison table from benchmark_results/*.json.

Usage: python -m mlflow_agent.compare [results_dir] [--md]
  --md  print a markdown table (for slides or a report)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text())
        res = data["results"]
        inj = [r for r in res if "injection" in r["tags"]]
        normal = [r for r in res if "injection" not in r["tags"]]

        def ok(rs):
            return sum(r["all_passed"] for r in rs)

        rows.append({
            "model": data["model"], "prompt": data["prompt_version"],
            "normal": f"{ok(normal)}/{len(normal)}", "inj": f"{ok(inj)}/{len(inj)}",
            "overall": ok(res) / len(res),
            "lat": sum(r["latency_s"] for r in res) / len(res),
        })
    rows.sort(key=lambda r: (-r["overall"], r["lat"]))
    top = rows[0]["overall"] if rows else 0
    for r in rows:
        r["top"] = r["overall"] == top
    return rows


def bar(p: float, width: int = 10) -> str:
    n = round(p * width)
    return "█" * n + "░" * (width - n)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = Path(args[0] if args else "benchmark_results")
    rows = load(d)
    if not rows:
        print(f"no result files in {d}")
        return
    head = ["", "Model", "Prompt", "Normal", "Injection", "Overall", "Sec/plan"]
    cells = [["★" if r["top"] else "", r["model"], r["prompt"], r["normal"], r["inj"],
              f"{bar(r['overall'])} {r['overall']:.0%}", f"{r['lat']:.1f}"] for r in rows]

    if "--md" in sys.argv:
        print("| " + " | ".join(head[1:]) + " |")
        print("|" + "|".join("---" for _ in head[1:]) + "|")
        for c in cells:
            print("| " + " | ".join(c[1:]) + " |")
        return

    widths = [max(len(x) for x in col) for col in zip(head, *cells)]

    def line(l, m, r):
        return l + m.join("─" * (w + 2) for w in widths) + r

    def row(c):
        return "│ " + " │ ".join(x.ljust(w) for x, w in zip(c, widths)) + " │"

    print("\nLLM + PROMPT BENCHMARK  (the raw LLM answer is scored, no retry, no fallback)\n")
    print(line("┌", "┬", "┐"))
    print(row(head))
    print(line("├", "┼", "┤"))
    for c in cells:
        print(row(c))
    print(line("└", "┴", "┘"))
    n_normal = rows[0]["normal"].split("/")[1]
    n_inj = rows[0]["inj"].split("/")[1]
    print(f"\n★ highest overall score (ties are all starred)")
    print(f"Normal    = {n_normal} regular reports, plan must pass every rule check")
    print(f"Injection = {n_inj} reports whose text tries to give orders to the LLM")
    print("Sec/plan  = average seconds for the LLM to answer on this machine\n")


if __name__ == "__main__":
    main()
