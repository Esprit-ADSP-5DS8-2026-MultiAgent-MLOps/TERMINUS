"""
Exécution des actions décidées. Chaque action produit un fichier : rien n'est invisible.

- open_incident / notify_data_team : rapport d'incident Markdown (cause, preuves, règles appliquées) ;
- increase_monitoring              : note de surveillance renforcée ;
- request_retraining               : retraining_request.json au format du contrat commun
                                     (common/schemas.py) + export des données labellisées récentes ;
- rollback_model                   : rollback_request.json (seulement après validation humaine).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..datasets import PACKAGE_ROOT
from .context import Case
from .schemas import AgentRun, Diagnosis


def _common_schemas():
    """Le contrat commun vit à la racine du repo (common/schemas.py)."""
    try:
        from common import schemas  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))
        from common import schemas  # type: ignore[import-not-found]
    return schemas


def incident_markdown(run: AgentRun, case: Case, title: str) -> str:
    d, p = run.diagnosis, run.policy
    lines = [f"# {title}", "",
             f"- **Projet** : {case.project} · **fenêtre** : {run.window_index} · **décision** : {run.id}",
             f"- **Cause** : `{d.root_cause}` · **confiance** : {d.confidence:.2f} · **accord** : {run.agreement:.0%}",
             f"- **Action** : `{p.final_action}` (proposée : `{p.proposed_action}`) · **mode** : {p.mode}",
             f"- **Décideur** : {run.decider} · **juge** : {run.judge or '—'}", "",
             "## Diagnostic", "", d.summary, "", f"_Pourquoi cette action :_ {d.rationale}", "",
             "## Preuves (vérifiées automatiquement)", "", "| fait | valeur | vérifié | ce que ça montre |", "|---|---|---|---|"]
    checks = {c.fact: c for c in (run.verification.checks if run.verification else [])}
    for e in d.evidence:
        ok = "oui" if checks.get(e.fact) and checks[e.fact].ok else "non"
        lines.append(f"| `{e.fact}` | {e.value:g} | {ok} | {e.claim} |")
    lines += ["", "## Règles de la policy", ""]
    lines += [f"- {'[ok]' if r.passed else '[bloquée]'} **{r.rule}** — {r.detail}" for r in p.rules]
    if run.verdicts:
        v = run.verdicts[-1]
        lines += ["", "## Avis du juge", "", f"**{v.verdict}** — {v.critique}", "",
                  " · ".join(f"{k} {s}/5" for k, s in v.scores.items())]
    return "\n".join(lines) + "\n"


def execute(action: str, run: AgentRun, case: Case, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    d: Diagnosis = run.diagnosis
    outputs: dict[str, str] = {}

    if action in ("open_incident", "notify_data_team", "request_retraining", "rollback_model"):
        title = {"open_incident": "Incident de monitoring", "notify_data_team": "Problème de qualité des données",
                 "request_retraining": "Demande de réentraînement", "rollback_model": "Demande de retour arrière"}[action]
        path = out_dir / f"incident_{run.id}.md"
        path.write_text(incident_markdown(run, case, title), encoding="utf-8")
        outputs["incident"] = str(path)

    if action == "increase_monitoring":
        path = out_dir / f"watch_{run.id}.json"
        path.write_text(json.dumps({"decision": run.id, "window": run.window_index, "reason": d.summary,
                                    "features": d.affected_features}, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["watch"] = str(path)

    if action == "request_retraining":
        s = _common_schemas()
        w = case.current
        new_data = case.export_new_data(str(out_dir / f"new_data_{run.id}.parquet")) if case.export_new_data else None
        perf = w.performance
        req = s.RetrainingRequest(
            model_uri=case.model_uri,
            reason="performance_drop" if case.signal_kinds() & {"performance_drop_realized", "concept_drift_suspected"}
            else "data_drift",
            drift=[s.FeatureDrift(feature=f.feature, method="psi", score=f.psi, drifted=True)
                   for f in w.feature_drift if f.drifted],
            current_performance=perf.realized if perf and perf.realized else None,
            new_data_path=new_data,
            diagnosis=f"[{d.root_cause}] {d.summary} — {d.rationale} (décision {run.id})",
        )
        path = out_dir / "retraining_request.json"
        path.write_text(req.model_dump_json(indent=2), encoding="utf-8")
        outputs["retraining_request"] = str(path)
        if new_data:
            outputs["new_data"] = new_data

    if action == "rollback_model":
        path = out_dir / f"rollback_request_{run.id}.json"
        path.write_text(json.dumps({"model_uri": case.model_uri, "decision": run.id, "reason": d.summary},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["rollback_request"] = str(path)
    return outputs
