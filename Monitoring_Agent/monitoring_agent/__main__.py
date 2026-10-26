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

Étape 3 (l'agent qui décide) :
    python -m monitoring_agent llm-check
    python -m monitoring_agent decide car_loan --at 5 [--rules] [--label-delay 3]
"""

from __future__ import annotations

import argparse
import json
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
    if not args.no_agent:
        from .agent.llm import decider_and_judge
        decider, judge = decider_and_judge()
        print(f"Agent : décideur {decider.name if decider else 'règles (aucune clé LLM)'} · "
              f"juge {judge.name if judge else 'aucun'}")
    uvicorn.run(create_app(target, check_interval_s=args.interval, agent=not args.no_agent),
                host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# Étape 3 : l'agent
# ---------------------------------------------------------------------------

ICONS = {"prepare": "📋", "investigate": "🔎", "diagnose": "🧠", "verify": "🧾", "judge": "⚖️ ",
         "policy": "🛡️ ", "await_human": "🙋", "act": "⚙️ "}


def print_run(run) -> None:
    print(f"\n── Décision {run.id} · fenêtre {run.window_index} · décideur {run.decider} · juge {run.judge or '—'}")
    for t in run.trace:
        print(f"  {ICONS.get(t.node, '•')} {t.node:<12} {t.ms:>8.0f} ms  {t.summary}")
        if t.node == "investigate" and t.detail.get("notes"):
            for line in t.detail["notes"].splitlines():
                print(f"  {'':>26}│ {line}")
    d, p = run.diagnosis, run.policy
    if d:
        print(f"\n  Diagnostic : {d.root_cause} · confiance {d.confidence:.2f} · accord {run.agreement:.0%}")
        print(f"  {d.summary}")
        print(f"  Pourquoi : {d.rationale}")
    if run.verification:
        for c in run.verification.checks:
            print(f"    {'✅' if c.ok else '❌'} {c.fact} = {c.claimed:g}" + ("" if c.ok else f" (réel : {c.actual})"))
    if run.verdicts:
        v = run.verdicts[-1]
        print(f"  Juge : {v.verdict} {v.scores} — {v.critique}")
    if p:
        print(f"\n  Policy : {p.proposed_action} -> {p.final_action} · mode {p.mode}")
        for r in p.rules:
            print(f"    {'✅' if r.passed else '⛔'} {r.rule} — {r.detail}")
        for reason in p.reasons:
            print(f"    → {reason}")
    print(f"\n  Statut : {run.status}" + (f" · erreur : {run.error}" if run.error else ""))
    for k, v in run.outputs.items():
        print(f"    {k}: {v}")


def cmd_decide(args) -> int:
    """L'agent sur une fenêtre d'un jeu NannyML (hors ligne, sans gateway)."""
    import pandas as pd

    from .agent.context import Case
    from .agent.graph import MonitoringAgent
    from .agent.llm import decider_and_judge
    from .analysis import Analyzer, make_windows
    from .datasets import load
    from .runtime.target import runs_dir

    ref, prod, spec = load(args.dataset, label_delay_windows=args.label_delay, window_size=args.window)
    analyzer = Analyzer(ref, spec.mapping, AnalyzerConfig(window_size=args.window))
    frames = make_windows(prod, args.window)
    upto = args.at if args.at is not None else len(frames) - 1
    reports = [analyzer.analyze_window(f, i) for i, f in enumerate(frames[:upto + 1])]

    def export(path: str) -> str | None:
        m = spec.mapping
        parts = [frames[r.window.index] for r in reports if r.dataset_drift.n_drifted and r.performance
                 and r.performance.labels_available]
        if not parts:
            return None
        pd.concat(parts)[m.features + [m.target]].to_parquet(path, index=False)
        return path

    decider, judge = (None, None) if args.rules else decider_and_judge()
    out = runs_dir() / "offline" / args.dataset
    from .memory.store import default_memory
    agent = MonitoringAgent(decider, judge, out_dir=out, memory=None if args.rules else default_memory())
    case = Case(project=args.dataset, mapping=spec.mapping, history=reports,
                model_uri=f"models:/{args.dataset}/1", export_new_data=export)
    w = case.current
    print(f"{spec.description}\nFenêtre {w.window.index} : sévérité {w.severity} · signaux "
          f"{', '.join(s.kind for s in w.signals) or 'aucun'}")
    run = agent.decide(case)
    print_run(run)
    if run.status == "awaiting_human":
        answer = args.answer
        if answer is None and sys.stdin.isatty():
            answer = "approve" if input("\n  Approuver cette action ? [o/N] ").strip().lower() in ("o", "oui", "y") else "reject"
        if answer:
            run = agent.resume(run.id, answer, by="cli")
            print_run(run)
    return 0 if run.status != "failed" else 1


def cmd_evaluate(args) -> int:
    from datetime import datetime

    from .agent.llm import decider_and_judge
    from .evaluation import evaluate, report_markdown
    from .runtime.target import runs_dir

    out = runs_dir() / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    methods = ["rules", "llm"] if args.method == "both" else [args.method]
    decider, judge = decider_and_judge() if "llm" in methods else (None, None)
    if "llm" in methods and decider is None:
        print("Aucun LLM configuré (Monitoring_Agent/.env) : seule la baseline à règles est évaluée.")
        methods = ["rules"]
    if args.redteam:
        from .evaluation import redteam
        print("\n== red team : consignes injectées dans les données ==")
        res = redteam(decider, judge, out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "redteam.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Agent complet : {res['agent_resisted']}/{res['cases']} attaques sans effet · "
              f"LLM seul : {res['llm_alone_resisted']}/{res['cases']}")
        return 0
    summaries = []
    for m in methods:
        print(f"\n== {m} ==")
        summaries.append(evaluate(m, decider, judge, per_dataset=args.per_dataset, out_dir=out))
    md = report_markdown(summaries)
    (out / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Résultats : {out}" + (f" · exemples de fine-tuning : {summaries[-1]['sft_examples']} (sft.jsonl)"
                                  if summaries[-1].get("sft_examples") else ""))
    return 0


def cmd_llm_check(_args) -> int:
    import time

    from .agent.llm import LLMError, decider_and_judge, list_models, structured
    from .agent.schemas import LLMModel

    class Ping(LLMModel):
        ok: bool
        sum: int

    decider, judge = decider_and_judge()
    ok = True
    for role, model in (("décideur", decider), ("juge", judge)):
        if model is None:
            print(f"{role:<9}: non configuré (clé absente de Monitoring_Agent/.env)")
            ok = False
            continue
        t0 = time.perf_counter()
        try:
            r = structured(model, [{"role": "user", "content": 'Réponds en JSON : {"ok": true, "sum": 17+25}'}], Ping)
            good = r.ok and r.sum == 42
            print(f"{role:<9}: {model.name} · JSON {'OK' if good else 'INCORRECT'} · {time.perf_counter() - t0:.1f} s")
            ok &= good
        except LLMError as exc:
            print(f"{role:<9}: {model.name} · ÉCHEC : {exc}")
            ok = False
    skip = ("guard", "image", "tts", "audio", "computer-use", "embedding", "whisper", "veo", "imagen", "live")
    for provider in ("groq", "gemini"):
        try:
            names = [n for n in list_models(provider) if not any(k in n for k in skip)]
            print(f"\nModèles {provider} utilisables ({len(names)}) : " + ", ".join(names))
        except Exception as exc:  # noqa: BLE001
            print(f"\nModèles {provider} : {exc}")
    print("\nChanger de modèle : MONITORING_DECIDER=groq:<modèle>, MONITORING_JUDGE=gemini:<modèle> dans .env")
    return 0 if ok else 1


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


def cmd_mcp(args) -> int:
    """Serveur MCP en stdio (Claude Desktop, MCP Inspector) sur un projet monitoré."""
    from .agent.mcp_server import server_for_target
    from .runtime.store import Store
    from .runtime.target import MonitoringTarget
    target = MonitoringTarget.load(args.project)
    server_for_target(target, Store(target.store_path)).run("stdio")
    return 0


def cmd_memory(args) -> int:
    from .memory.cards import from_incident_data, from_scenarios
    from .memory.store import IncidentMemory, evaluate_retrieval, memory_dir
    mem = IncidentMemory(memory_dir())
    scen = Path(args.dir) / "scenarios.jsonl"
    if args.step == "build":
        if mem.count() and not args.append:
            print(f"Mémoire déjà construite ({mem.count()} fiches). --append pour ajouter.")
            return 0
        from .datasets import data_dir
        cards = from_scenarios(scen) + from_incident_data(data_dir())
        ids = mem.add(cards)
        by = {}
        for c in cards:
            by[c.source] = by.get(c.source, 0) + 1
        print(f"{len(ids)} fiches ajoutées ({by}) · embedder {mem.embedder.name} -> {memory_dir()}")
        return 0
    if args.step == "eval":
        rows = [json.loads(x) for x in scen.read_text(encoding="utf-8").splitlines() if x.strip()]
        test = [r for r in rows if r["split"] == "test"][: args.limit or None]
        valid = [r for r in rows if r["split"] == "valid"]
        res = evaluate_retrieval(mem, test, k=3, validation=valid)
        (Path(args.dir) / "eval_retrieval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Poids du texte calibré sur la validation (mc_prepaid) : {res['text_weight']:g}")
        print(f"{res['queries']} requêtes (bases de test, jamais en mémoire), k = 3")
        for mode, m in res["modes"].items():
            print(f"  {mode:<28} hit@3 {m['hit@k']:.0%} · précision@3 {m['precision@k']:.0%} · MRR {m['mrr']:.2f} "
                  f"· sans résultat {m['no_result']:.0%}")
        return 0
    if args.step == "search":
        # Démo : un scénario de TEST (jamais en mémoire) -> les 3 cas retrouvés par chaque méthode
        from .memory.store import _queries
        rows = [json.loads(x) for x in scen.read_text(encoding="utf-8").splitlines() if x.strip()]
        test = [r for r in rows if r["split"] == "test"]
        sc = next((r for r in test if r["id"] == args.id), None) if args.id else test[0]
        if sc is None:
            print(f"scénario inconnu : {args.id} (ex. {test[0]['id']})")
            return 1
        print(f"Requête : {sc['id']} · injection {sc['injection']} · signaux {', '.join(sc['signals']) or 'aucun'}")
        print(f"Bonne réponse (oracle) : {sc['oracle']['root_cause']} -> {sc['oracle']['action']}")
        q_text, q_sig, _ = _queries(mem, [sc])[0]
        print(f"Signature : {q_sig}")
        for mode, w in (("text", None), ("sig", None), ("hybrid", None)):
            hits = mem.search_vectors(q_text, q_sig, "ml", k=3, mode=mode, text_weight=w)
            label = f"hybride (texte ×{mem.text_weight:g})" if mode == "hybrid" else mode
            print(f"\n  {label}")
            for h in hits:
                ok = "✓" if h["root_cause"] == sc["oracle"]["root_cause"] else "✗"
                ts = f"{h['text_sim']:.2f}" if h["text_sim"] else " —  "    # — : hors des 20 meilleurs candidats
                ss = f"{h['sig_sim']:.2f}" if h["sig_sim"] else " —  "
                print(f"    {ok} {h['id']} texte {ts} · chiffres {ss} — {h['title']} "
                      f"-> {h['root_cause']} / {h['action']}")
        return 0
    if args.step == "show":
        for c in mem.all()[: args.limit or 20]:
            print(f"{c.id} [{c.family}/{c.source}] {c.title} -> {c.root_cause} / {c.action} ({c.outcome})")
        return 0
    return 1


def cmd_langfuse(args) -> int:
    from . import observability as obs
    lf = obs.client()
    if lf is None:
        print("Langfuse non configuré : LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL dans .env")
        return 1
    if args.step == "check":
        print("Connexion Langfuse :", "OK" if lf.auth_check() else "ÉCHEC")
    elif args.step == "push-prompts":
        print("Prompts versionnés :", ", ".join(obs.push_prompts()))
    elif args.step == "push-dataset":
        path = Path(args.dir) / "data" / "test_cases.jsonl"
        cases = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"Dataset agent4-diagnosis-test : {obs.push_dataset(cases)} cas")
    return 0


def cmd_mcp_check(args) -> int:
    """Client MCP de démonstration : lance le serveur en stdio (vrai transport JSON-RPC) et l'interroge."""
    import asyncio

    from mcp import Client, StdioServerParameters

    from .agent.mcp_server import result_dict
    from .datasets import PACKAGE_ROOT

    async def run() -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "monitoring_agent", "mcp", args.project],
                                       cwd=str(PACKAGE_ROOT))
        async with Client(params) as c:
            info = c.server_info
            print(f"Serveur : {info.name if info else '?'} · protocole {c.protocol_version}")
            tools = (await c.list_tools()).tools
            print(f"\n{len(tools)} outils (tools/list) :")
            for t in tools:
                ro = getattr(getattr(t, "annotations", None), "read_only_hint", None)
                ro = ro if ro is not None else getattr(getattr(t, "annotations", None), "readOnlyHint", None)
                print(f"  - {t.name}{'  [lecture seule]' if ro else ''} : {(t.description or '').splitlines()[0]}")
            res = (await c.list_resources()).resources
            tpl = (await c.list_resource_templates()).resource_templates if hasattr(c, "list_resource_templates") else []
            print(f"\nRessources : {', '.join(str(r.uri) for r in res)}"
                  + (f" · modèles : {', '.join(getattr(t, 'uri_template', '') for t in tpl)}" if tpl else ""))
            prompts = (await c.list_prompts()).prompts
            print(f"Prompts : {', '.join(p.name for p in prompts)}")
            status = await c.read_resource("monitoring://status")
            text = status.contents[0].text if status.contents else "{}"
            print("\nmonitoring://status (extrait) :\n  " + json.loads(text)["summary"].replace("\n", "\n  ")[:600])
            hist = result_dict(await c.call_tool("performance_history", {"last_n": 4}))
            print(f"\ntools/call performance_history -> {json.dumps(hist, ensure_ascii=False)[:400]}")
            health = result_dict(await c.call_tool("service_health", {}))
            print(f"tools/call service_health -> {json.dumps(health, ensure_ascii=False)[:300]}")

    asyncio.run(run())
    return 0


FT_DIR = "runs/finetune"
FT_SPLITS = {"train": ["synthetic_sample", "employment_MA", "bank", "bike", "california", "california_reg"],
             "valid": ["mc_prepaid"], "test": ["car_loan", "car_price"]}


def cmd_finetune(args) -> int:
    from pathlib import Path

    from .finetune import bases
    out = Path(args.dir)
    scen, teach_out, data = out / "scenarios.jsonl", out / "teacher.jsonl", out / "data"
    if args.step == "bases":
        for name, line in bases.available().items():
            print(f"  {name:<17} {line}")
        return 0
    if args.step == "scenarios":
        from .finetune.scenarios import generate
        per = {b: args.per_base for b in FT_SPLITS["train"]}
        per |= {b: max(40, args.per_base // 2) for b in FT_SPLITS["valid"] + FT_SPLITS["test"]}
        stats = generate(scen, per, workers=args.workers)
        kept = sum(v for (_b, _i, ok), v in stats.items() if ok)
        print(f"{kept} scénarios acceptés / {sum(stats.values())} générés -> {scen}")
        return 0
    if args.step == "teach":
        from .finetune.teacher import teach
        teach(scen, teach_out, args.teachers, args.max_calls)
        return 0
    if args.step == "build":
        from .finetune.build import build
        stats = build(scen, teach_out, data, templates=not args.no_templates, epochs=args.epochs)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"\nEntraînement (Mac Apple Silicon) :\n  mlx_lm.lora --config {data / 'lora.yaml'}")
        return 0
    if args.step == "eval":
        from .finetune.evaluate import evaluate_model, report
        results = []
        for spec in args.models.split(","):
            print(f"\n== {spec}")
            results.append(evaluate_model(spec.strip(), data / "test_cases.jsonl", args.limit))
        md = report(results)
        (out / "eval_slm.md").write_text(md, encoding="utf-8")
        (out / "eval_slm.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + md)
        return 0
    return 1


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
    g.add_argument("--no-agent", action="store_true", help="monitoring seul, sans l'agent qui décide")
    g.set_defaults(func=cmd_gateway)

    d = sub.add_parser("decide", help="l'agent décide sur une fenêtre d'un jeu NannyML (hors ligne)")
    d.add_argument("dataset")
    d.add_argument("--at", type=int, help="index de la fenêtre (défaut : la dernière)")
    d.add_argument("--window", type=int, default=5000)
    d.add_argument("--label-delay", type=int, default=0)
    d.add_argument("--rules", action="store_true", help="décideur à règles, sans LLM (baseline)")
    d.add_argument("--answer", choices=["approve", "reject"], help="réponse si un humain est requis")
    d.set_defaults(func=cmd_decide)

    sub.add_parser("llm-check", help="vérifie les clés et les modèles LLM").set_defaults(func=cmd_llm_check)

    e = sub.add_parser("evaluate", help="évaluation chiffrée : règles vs LLM seul vs agent complet")
    e.add_argument("--method", choices=["rules", "llm", "both"], default="both")
    e.add_argument("--per-dataset", type=int, default=None,
                   help="nombre de fenêtres dégradées par jeu (défaut : 5 ; 2 suffit pour un essai LLM rapide)")
    e.add_argument("--redteam", action="store_true", help="injection de consignes par les données (prompt injection)")
    e.set_defaults(func=cmd_evaluate)

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

    mc = sub.add_parser("mcp", help="serveur MCP (stdio) d'un projet monitoré, pour Claude Desktop / MCP Inspector")
    mc.add_argument("project")
    mc.set_defaults(func=cmd_mcp)

    lfp = sub.add_parser("langfuse", help="LLMOps : vérifier la connexion, versionner les prompts, pousser le dataset")
    lfp.add_argument("step", choices=["check", "push-prompts", "push-dataset"])
    lfp.add_argument("--dir", default="runs/finetune")
    lfp.set_defaults(func=cmd_langfuse)

    me = sub.add_parser("memory", help="mémoire d'incidents (RAG hybride) : construire, évaluer, afficher")
    me.add_argument("step", choices=["build", "eval", "show", "search"])
    me.add_argument("--id", help="search : identifiant d'un scénario de test (défaut : le premier)")
    me.add_argument("--dir", default="runs/finetune", help="dossier contenant scenarios.jsonl")
    me.add_argument("--append", action="store_true")
    me.add_argument("--limit", type=int)
    me.set_defaults(func=cmd_memory)

    mck = sub.add_parser("mcp-check", help="démo client MCP : liste et appelle les outils du serveur (stdio)")
    mck.add_argument("project")
    mck.set_defaults(func=cmd_mcp_check)

    ft = sub.add_parser("finetune", help="dataset de fine-tuning du diagnostiqueur local (Qwen3-4B, MLX)")
    ft.add_argument("step", choices=["bases", "scenarios", "teach", "build", "eval"])
    ft.add_argument("--dir", default=FT_DIR)
    ft.add_argument("--per-base", type=int, default=220, help="scenarios : exemples acceptés par base d'entraînement")
    ft.add_argument("--teachers", help="teach : liste fournisseur:modèle séparée par des virgules")
    ft.add_argument("--max-calls", type=int, default=400, help="teach : budget d'appels LLM pour cette session")
    ft.add_argument("--no-templates", action="store_true", help="build : seulement les exemples du professeur")
    ft.add_argument("--epochs", type=int, default=2)
    ft.add_argument("--workers", type=int, help="scenarios : processus en parallèle (défaut : nb de cœurs - 1)")
    ft.add_argument("--models", default="mlx:default_model", help="eval : modèles à comparer (virgules)")
    ft.add_argument("--limit", type=int)
    ft.set_defaults(func=cmd_finetune)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
