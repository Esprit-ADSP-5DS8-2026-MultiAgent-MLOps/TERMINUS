"""Judge (évaluateur intégré au Repair Agent). Fonction : (projet source, rapport de réparation [, rapport du Validator]) -> JudgeVerdict scellé.

Jamais de décision d'admission (rôle exclusif du Validator, après revalidation). Échec fermé : en cas d'erreur, le verdict n'est jamais ACCEPT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from agentic_core.loop import AgentSpec, LoopConfig, run_loop
from agentic_core.progress import NullProgress
from agentic_core.schemas import (CRITERIA, Criterion, Failure, FailureKind, JudgeVerdict, RepairReport, RunStatus, ValidatorReport,
                                  chain_from, load_report, new_id, now_iso, seal, verify_seal)
from validator.handoff import compute_project_hash

from .prompts import PROMPT
from .schemas import VerdictSub
from .tools import JudgeToolBox

ALLOWED_NEXT = {"ACCEPT": {"REVALIDATE"}, "REVISE": {"RETRY", "ESCALATE"}, "REJECT": {"ROLLBACK", "ESCALATE"}}


@dataclass
class JudgeConfig:
    loop: LoopConfig = field(default_factory=lambda: LoopConfig(max_steps=10, token_budget=50_000))
    require_distinct_model: bool = False      # True : même modèle que le Repair => échec fermé (ESCALATE) au lieu d'un simple avertissement




def _checker(tb: JudgeToolBox):
    def check(args):
        try:
            v = VerdictSub.model_validate(args)
        except ValidationError as e:
            return None, [f"format invalide : {err['loc']} {err['msg']}"[:160] for err in e.errors()[:5]]
        p = []
        names = [c.name for c in v.criteria]
        if sorted(names) != sorted(CRITERIA):
            p.append(f"il faut exactement les 5 critères {list(CRITERIA)} (reçu {names})")
        for c in v.criteria:
            if not c.evidence:
                p.append(f"critère {c.name} sans evidence")
            elif not tb.is_grounded(c.evidence):
                p.append(f"critère {c.name} : evidence introuvable dans les sorties d'outils (copie un extrait exact d'un diff ou d'un test)")
        if any(c.evidence and not tb.is_grounded(c.evidence) for c in v.criteria) and (h := tb.evidence_hints()):
            p.append("Extraits VALIDES à recopier tels quels dans `evidence` : " + " | ".join(h))
        if not tb.diff_examined:
            p.append("tu n'as pas examiné le diff (list_changes / get_diff) avant de juger")
        if v.verdict == "ACCEPT" and any(c.result == "FAIL" for c in v.criteria):
            p.append("ACCEPT impossible avec un critère FAIL")
        if v.next_action not in ALLOWED_NEXT[v.verdict]:
            p.append(f"next_action {v.next_action} incompatible avec {v.verdict} (autorisé : {sorted(ALLOWED_NEXT[v.verdict])})")
        if v.verdict == "ACCEPT":
            b = tb.blockers_for_accept()
            if b:
                p.append("ACCEPT interdit par le protocole : " + " ; ".join(b))
        return v, p[:10]
    return check


def run_judge(project, repair_report, client, validator_report=None, cfg: JudgeConfig | None = None, *, workspace=None, progress=None) -> JudgeVerdict:
    cfg, progress = cfg or JudgeConfig(), progress or NullProgress()
    t0 = time.time()
    try:
        rr = load_report(repair_report)
    except Exception as e:
        return seal(JudgeVerdict(verdict_id=new_id(), created_at=now_iso(), repair_id="?", verdict="REVISE", next_action="ESCALATE",
                                 run_status=RunStatus.FAILED_CLOSED, failure=Failure(kind=FailureKind.INPUT_INVALID, message=f"rapport illisible : {type(e).__name__}")))
    base = dict(verdict_id=new_id(), created_at=now_iso(), repair_id=getattr(rr, "repair_id", "?"), project=getattr(rr, "project", ""),
                lineage=chain_from(rr if isinstance(rr, RepairReport) else None), repair_model=(getattr(rr, "llm", None) or {}).get("model"))

    def closed(kind, msg, **extra):
        return seal(JudgeVerdict(**base, verdict="REVISE", next_action="ESCALATE", run_status=RunStatus.FAILED_CLOSED,
                                 failure=Failure(kind=kind, message=msg), judge_model=getattr(client, "model", None),
                                 reason="échec fermé : jamais de verdict favorable par défaut", **extra))
    if not isinstance(rr, RepairReport) or not verify_seal(rr):
        return closed(FailureKind.INPUT_INVALID, "rapport de réparation invalide ou modifié (sceau)")
    src = Path(project).resolve()
    ws = Path(workspace or rr.workspace_path or "")
    if not src.is_dir() or not ws.is_dir():
        return closed(FailureKind.INPUT_INVALID, "projet source ou copie corrigée introuvable")
    if rr.source_project_hash and compute_project_hash(src)[0] != rr.source_project_hash:
        return closed(FailureKind.INPUT_INVALID, "le projet source ne correspond pas à la réparation (empreinte)")
    if rr.repaired_project_hash and compute_project_hash(ws)[0] != rr.repaired_project_hash:
        return closed(FailureKind.INPUT_INVALID, "la copie corrigée a changé depuis la réparation (empreinte)")
    vr = load_report(validator_report) if validator_report is not None else None
    judge_model = getattr(client, "model", None)
    independent = (judge_model != base["repair_model"]) if (judge_model and base["repair_model"]) else None
    if cfg.require_distinct_model and independent is False:
        return closed(FailureKind.INPUT_INVALID, "le Judge utilise le même modèle que le Repair (require_distinct_model)", independent_model=False)

    tb = JudgeToolBox(src, ws, rr, vr if isinstance(vr, ValidatorReport) else None)
    spec = AgentSpec(name="judge", system_prompt=PROMPT,
                     user_message=(f"Évalue la réparation {rr.repair_id} du projet « {rr.project} » (tentative {rr.attempt}, statut du Repair : {rr.status}). "
                                   "Examine d'abord list_changes et get_diff, puis les tests, puis dépose submit_verdict."),
                     toolbox=tb, submit_tool="submit_verdict", check_submission=_checker(tb),
                     nudge="Utilise les outils (list_changes, get_diff, get_test_results…) puis appelle submit_verdict.")
    res = run_loop(client, spec, cfg.loop, progress)
    v: VerdictSub | None = res.submission
    # avec des LLM de secours, le modèle qui a RÉELLEMENT répondu peut différer du modèle principal : on le relit APRÈS la boucle
    judge_model = getattr(client, "model", None) or judge_model
    independent = (judge_model != base["repair_model"]) if (judge_model and base["repair_model"]) else None
    if cfg.require_distinct_model and independent is False:
        return closed(FailureKind.INPUT_INVALID, "après bascule, le Judge a utilisé le même modèle que le Repair (require_distinct_model)", independent_model=False)
    kw = dict(base, judge_model=judge_model, independent_model=independent, llm={"model": judge_model, **res.stats, "tool_calls": len(tb.calls)})
    if v is None:
        return seal(JudgeVerdict(**kw, verdict="REVISE", next_action="ESCALATE", run_status=RunStatus.FAILED_CLOSED,
                                 failure=res.failure or Failure(kind=FailureKind.INVALID_SUBMISSION, message="aucun verdict valide"),
                                 reason="échec fermé : jamais de verdict favorable par défaut"))
    crits = [Criterion(**c.model_dump(), grounded=tb.is_grounded(c.evidence)) for c in v.criteria]
    verdict, nxt, reason, over = v.verdict, v.next_action, v.reason, None
    blockers = tb.blockers_for_accept() if verdict == "ACCEPT" else []
    if verdict == "ACCEPT" and (blockers or any(c.result == "FAIL" for c in crits) or any(not c.grounded for c in crits) or res.problems):
        # le LLM a insisté malgré les refus : l'infrastructure tranche, en défaveur de la réparation
        over = "; ".join(blockers + res.problems)[:400] or "ACCEPT non justifié"
        verdict = "REVISE"
        nxt = "ESCALATE" if tb.flags or "approbation" in over else "RETRY"
        reason = f"[forcé par invariant] {over}. Raison initiale : {v.reason}"
    if res.failure:                                   # soumission partielle d'une boucle interrompue : jamais favorable
        verdict, nxt = ("REVISE", "ESCALATE") if verdict == "ACCEPT" else (verdict, nxt)
    concerns = list(v.remaining_concerns)
    if independent is False:
        concerns.append("le Judge utilise le même modèle que le Repair : risque de biais d'auto-approbation")
    return seal(JudgeVerdict(**kw, verdict=verdict, criteria=crits, evidence_examined=v.evidence_examined, remaining_concerns=concerns,
                             reason=reason, next_action=nxt, overridden_by_invariant=over,
                             run_status=RunStatus.FAILED_CLOSED if res.failure else RunStatus.OK, failure=res.failure))
