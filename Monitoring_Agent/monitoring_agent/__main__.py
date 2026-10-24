"""
Ligne de commande de l'Agent 4.

Étape 1 (hors ligne, jeux NannyML) :
    python -m monitoring_agent datasets
    python -m monitoring_agent analyze car_loan [--label-delay 3] [--out runs/report.json]

Étape 2 (en ligne, devant l'API de l'Agent 3) :
    python -m monitoring_agent init ../eval/projects/credit_risk --upstream http://127.0.0.1:8000 --name credit_risk
    python -m monitoring_agent gateway credit_risk
    python -m monitoring_agent simulate credit_risk --scenario covariate
    python -m monitoring_agent provision credit_risk
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


INFRA = Path(__file__).resolve().parent.parent / "infra"


def cmd_init(args) -> int:
    from .runtime.target import build_target
    print(f"Construction de la référence via {args.upstream}/predict ...")
    target, info = build_target(args.project_dir, args.upstream, args.name, args.window)
    m = target.mapping
    print(f"  projet       : {target.project} ({m.task}, métrique {m.primary_metric}, seuil {m.min_threshold})")
    print(f"  features     : {len(m.features)}  ·  cible : {m.target}")
    print(f"  référence    : {info['scored']}/{info['rows']} lignes notées par le modèle déployé"
          + (f" ({info['failed']} échecs)" if info["failed"] else ""))
    print(f"  écrit        : {target.dir}/monitoring.json, reference.parquet")
    if not args.no_provision:
        return cmd_provision(argparse.Namespace(project=target.project))
    return 0


def cmd_provision(args) -> int:
    from .provisioning.grafana import write_dashboard
    from .provisioning.prometheus import check_rules, write_rules
    from .runtime.target import MonitoringTarget
    target = MonitoringTarget.load(args.project)
    rules = write_rules(target, INFRA / "prometheus" / "rules")
    ok, out = check_rules(rules)
    status = {True: "valide (promtool)", False: "INVALIDE", None: "non vérifié"}[ok]
    print(f"Règles d'alerte : {rules}  ->  {status}")
    if ok is False or ok is None:
        print("  " + out.replace("\n", "\n  "))
    dash = write_dashboard(target, INFRA / "grafana" / "dashboards")
    print(f"Dashboard Grafana : {dash}")
    print("Recharger Prometheus s'il tourne déjà : curl -X POST http://localhost:9090/-/reload")
    return 0 if ok is not False else 1


def cmd_gateway(args) -> int:
    import logging

    import uvicorn

    from .runtime.gateway import create_app
    from .runtime.target import MonitoringTarget
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s · %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # une ligne par requête noierait les fenêtres
    target = MonitoringTarget.load(args.project)
    if args.upstream:
        target.upstream_url = args.upstream.rstrip("/")
    print(f"Gateway {target.project} : http://{args.host}:{args.port}  ->  {target.upstream_url}")
    uvicorn.run(create_app(target, check_interval_s=args.interval), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_simulate(args) -> int:
    import asyncio

    import pandas as pd

    from .runtime.simulator import simulate
    from .runtime.target import MonitoringTarget
    target = MonitoringTarget.load(args.project)
    data = pd.read_parquet(target.reference_path).drop(columns=["y_pred", "y_pred_proba"], errors="ignore")
    print(f"Scénario « {args.scenario} » : {args.n} requêtes ({args.warmup} normales d'abord), "
          f"{args.rate:.0f} req/s, labels envoyés avec {args.label_delay} requêtes de retard")
    res = asyncio.run(simulate(args.gateway, data, target.mapping.target, args.scenario, args.n, args.warmup,
                               args.rate, args.label_delay, args.seed, ramp=args.ramp))
    print(f"Envoyées {res.sent} · réussies {res.ok} · erreurs {res.errors} · labels {res.labels_sent} "
          f"· {res.seconds:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m monitoring_agent", description="Agent 4 : monitoring")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="liste les jeux de données disponibles").set_defaults(func=cmd_datasets)

    i = sub.add_parser("init", help="prépare le monitoring d'un projet déployé (référence + configs)")
    i.add_argument("project_dir", help="dossier du projet (mlops.yaml + data.csv)")
    i.add_argument("--upstream", default="http://127.0.0.1:8000", help="URL de l'API du modèle (Agent 3)")
    i.add_argument("--name", help="nom du projet (défaut : nom du dossier)")
    i.add_argument("--window", type=int, default=200, help="taille d'une fenêtre (prédictions)")
    i.add_argument("--no-provision", action="store_true")
    i.set_defaults(func=cmd_init)

    pv = sub.add_parser("provision", help="génère et valide les règles Prometheus et le dashboard Grafana")
    pv.add_argument("project")
    pv.set_defaults(func=cmd_provision)

    g = sub.add_parser("gateway", help="lance la gateway de monitoring devant l'API du modèle")
    g.add_argument("project")
    g.add_argument("--port", type=int, default=8100)
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--upstream", help="remplace l'URL de l'API du modèle")
    g.add_argument("--interval", type=float, default=2.0, help="période du moniteur (s)")
    g.set_defaults(func=cmd_gateway)

    s = sub.add_parser("simulate", help="envoie du trafic de production simulé à la gateway")
    s.add_argument("project")
    s.add_argument("--scenario", default="normal", choices=["normal", "covariate", "quality", "concept"])
    s.add_argument("--gateway", default="http://127.0.0.1:8100")
    s.add_argument("--n", type=int, default=1600)
    s.add_argument("--warmup", type=int, default=600, help="requêtes normales avant le scénario")
    s.add_argument("--ramp", type=int, default=200, help="requêtes pour atteindre la pleine intensité")
    s.add_argument("--rate", type=float, default=20.0, help="requêtes par seconde")
    s.add_argument("--label-delay", type=int, default=200, help="retard des labels (en requêtes)")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_simulate)

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
