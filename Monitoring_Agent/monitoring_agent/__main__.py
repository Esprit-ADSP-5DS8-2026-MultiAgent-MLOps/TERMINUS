"""
Ligne de commande de l'Agent 4.

    python -m monitoring_agent datasets
    python -m monitoring_agent analyze car_loan
    python -m monitoring_agent analyze car_loan --label-delay 3 --out runs/car_loan_report.json
    python -m monitoring_agent analyze car_loan_dq --window 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import AnalyzerConfig


def _fmt(v, digits=3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def cmd_datasets(_args) -> int:
    from .datasets import DATASETS, data_dir
    print(f"Dossier de données : {data_dir()}\n")
    for name, spec in DATASETS.items():
        print(f"  {name:<12} {spec.description}")
    return 0


def cmd_analyze(args) -> int:
    from .analysis import Analyzer
    from .datasets import load

    cfg = AnalyzerConfig(window_size=args.window)
    t0 = time.perf_counter()
    reference, production, spec = load(args.dataset, with_labels=not args.no_labels,
                                       label_delay_windows=args.label_delay, window_size=args.window)
    analyzer = Analyzer(reference, spec.mapping, cfg)
    report = analyzer.analyze(production, project=args.dataset)
    elapsed = time.perf_counter() - t0

    prof = analyzer.profile
    metric = spec.mapping.primary_metric
    print(f"\n{spec.description}")
    print(f"référence : {prof.n_rows} lignes · production : {len(production)} lignes · fenêtres de {args.window}")
    if prof.metric_mean is not None:
        print(f"{metric} sur la référence : {prof.metric_mean:.4f} ± {prof.metric_std:.4f} (par fenêtre)\n")

    header = f"{'#':>2}  {'début':<19}  {'drift':>5}  {'features en drift':<44} {'DQ':<9} {metric+' estimé':>12} {metric+' réel':>10}  {'sévérité':<8} agent"
    print(header)
    print("─" * len(header))
    for w in report.windows:
        p = w.performance
        est = p.estimated.get(metric) if p and p.estimated else None
        real = p.realized.get(metric) if p and p.realized else None
        flag_est = "▼" if p and p.estimated_degraded else " "
        flag_real = "▼" if p and p.realized_degraded else " "
        dq = ",".join(sorted({q.kind.split("_")[0] for q in w.data_quality if q.severity in ("warning", "critical")})) or "·"
        feats = ", ".join(w.dataset_drift.drifted_features) or "·"
        print(f"{w.window.index:>2}  {(w.window.start or '')[:19]:<19}  "
              f"{w.dataset_drift.n_drifted:>2}/{w.dataset_drift.n_features:<2}  {feats[:44]:<44} {dq[:9]:<9} "
              f"{_fmt(est, 4):>11}{flag_est} {_fmt(real, 4):>9}{flag_real}  {w.severity:<8} {'OUI' if w.trigger_agent else ''}")

    s = report.summary
    print(f"\nRésumé : {s['n_triggered']}/{s['n_windows']} fenêtres réveilleraient l'agent · "
          f"1er drift : fenêtre {s['first_drift_window']} · 1re baisse estimée : {s['first_estimated_drop_window']} · "
          f"1re baisse mesurée : {s['first_realized_drop_window']}")
    print(f"Features les plus souvent en drift : {', '.join(s['most_drifted_features']) or '—'}")
    print(f"Analyse en {elapsed:.1f} s")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Rapport écrit : {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m monitoring_agent", description="Agent 4 : monitoring")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="liste les jeux de données disponibles").set_defaults(func=cmd_datasets)

    a = sub.add_parser("analyze", help="analyse une période de production fenêtre par fenêtre")
    a.add_argument("dataset")
    a.add_argument("--window", type=int, default=5000, help="taille d'une fenêtre (lignes)")
    a.add_argument("--label-delay", type=int, default=0, help="les N dernières fenêtres n'ont pas encore de labels")
    a.add_argument("--no-labels", action="store_true", help="aucun label de production (estimation seule)")
    a.add_argument("--out", help="chemin du monitoring_report.json")
    a.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
