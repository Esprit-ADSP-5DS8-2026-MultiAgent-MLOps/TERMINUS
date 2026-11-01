"""
Adaptateur de la chaîne Validator v2 -> Repair -> Judge (Sarra) pour le nœud « repair » de l'orchestrateur.

    REAL["repair"] = "repair_agent.orchestrator_adapter:run"

Quand le validateur v1 (barrière d'entrée du pipeline) bloque un projet, l'orchestrateur appelle run() :
1. validator_agent (v2) diagnostique le projet : contrôles, problèmes classés, statut calculé par le moteur de politique ;
2. repair_agent corrige une COPIE dans le workdir, sous le même nom de dossier (le projet d'origine n'est jamais
   modifié), et relance les tests ;
3. le Judge (modèle d'une autre famille) rend son verdict sur la réparation ;
4. l'orchestrateur revalide la copie avec le v1 ; à la tentative suivante, le verdict REVISE du Judge est transmis
   au Repair (son mode « --attempt 2 --feedback »).

Les trois rapports scellés (validator_v2.json, repair.json, judge.json) restent dans le workdir de la tentative.
Configuration : agents/validator/.env (GROQ_API_KEY, GROQ_MODEL_<AGENT>...), appliqué seulement pendant la chaîne.
REPAIR_ENV (venv | current, défaut venv), REPAIR_SANDBOX (auto | docker | subprocess, défaut auto).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from common.envscope import agent_env
from common.schemas import (
    AgentInput,
    AgentResult,
    AgentStatus,
    RepairChange,
    RepairResult,
    write_output,
)
from common.schemas import (
    JudgeVerdict as ContractVerdict,
)

AGENT_DIR = Path(__file__).resolve().parents[1]        # agents/validator/


def _fail(started: float, reasons: list[str], outputs: dict | None = None, warnings: list | None = None) -> AgentResult:
    return AgentResult(agent="repair", status=AgentStatus.WARNING, can_proceed=False, outputs=outputs or {},
                       reasons=reasons, warnings=warnings or [], duration_s=round(time.monotonic() - started, 3))


def _previous_verdict(workdir: Path) -> str | None:
    """Verdict REVISE du Judge à la tentative précédente : il guide la nouvelle réparation."""
    try:
        n = int(workdir.name.rsplit("_", 1)[-1])
    except ValueError:
        return None
    prev = workdir.parent / f"attempt_{n - 1}" / "judge.json"
    if prev.is_file() and json.loads(prev.read_text(encoding="utf-8")).get("verdict") == "REVISE":
        return str(prev)
    return None


def run(inp: AgentInput) -> AgentResult:
    started = time.monotonic()
    workdir = Path(inp.workdir)
    project = Path(inp.project_dir)
    attempt = int(workdir.name.rsplit("_", 1)[-1]) if workdir.name.startswith("attempt_") else 1
    feedback = _previous_verdict(workdir)
    paths = {"validator_v2": workdir / "validator_v2.json", "repair_report": workdir / "repair.json",
             "judge_verdict": workdir / "judge.json"}

    with agent_env(AGENT_DIR / ".env"):
        from agentic_core.llm_client import build_client
        from validator_agent.agent import ValidatorConfig, run_validator

        from repair_agent.agent import RepairConfig, run_repair
        from repair_agent.judge.evaluator import run_judge

        env = os.environ.get("REPAIR_ENV") or "venv"
        sandbox = os.environ.get("REPAIR_SANDBOX") or "auto"

        # 1. diagnostic du validateur v2
        vr = run_validator(project, build_client("validator"), cfg=ValidatorConfig(env=env, sandbox=sandbox))
        paths["validator_v2"].write_text(vr.model_dump_json(indent=2), encoding="utf-8")
        outputs = {"validator_v2": str(paths["validator_v2"])}
        summary = f"validateur v2 : {vr.status.value} ({len(vr.issues)} problème(s))"
        if vr.run_status.value != "OK":
            return _fail(started, [f"{summary}, analyse interrompue : "
                                   f"{vr.failure.message if vr.failure else vr.run_status.value}"[:400]], outputs)
        if vr.next_action != "REPAIR":
            why = {"PROCEED": "le v2 accepte le projet alors que le v1 le bloque : écart à examiner par un humain",
                   "STOP": "projet rejeté (hors périmètre ou non sûr) : pas de réparation automatique",
                   "HUMAN_REVIEW": "décision humaine requise avant toute réparation"}.get(vr.next_action, vr.next_action)
            return _fail(started, [f"{summary} : {why}"], outputs)

        # 2. réparation d'une copie dans le workdir (attempt > 1 : avec le verdict REVISE précédent)
        # la copie garde le nom du projet : Agent 1 en tire project_name (modèle MLflow, image Docker)
        rr = run_repair(project, str(paths["validator_v2"]), str(workdir / project.name), build_client("repair"),
                        RepairConfig(env=env, sandbox=sandbox), attempt=attempt, judge_feedback=feedback)
        paths["repair_report"].write_text(rr.model_dump_json(indent=2), encoding="utf-8")
        outputs["repair_report"] = str(paths["repair_report"])
        if rr.status not in ("REPAIRED", "PARTIAL") or rr.next_action != "JUDGE" or not rr.workspace_path:
            unresolved = "; ".join(f"{u.issue_id} : {u.reason}" for u in rr.unresolved_issues[:3])
            return _fail(started, [f"réparation {rr.status} : {rr.summary or unresolved or 'aucune correction possible'}"[:400]],
                         outputs)

        # 3. verdict du Judge
        jv = run_judge(project, str(paths["repair_report"]), build_client("judge"),
                       validator_report=str(paths["validator_v2"]))
        paths["judge_verdict"].write_text(jv.model_dump_json(indent=2), encoding="utf-8")
        outputs["judge_verdict"] = str(paths["judge_verdict"])

    issues = {i.issue_id: i.check_id for i in vr.issues}
    changes = [RepairChange(file=m.file, finding_code=issues.get(m.issue_id or "", m.issue_id or "?"),
                            description=m.reason or m.action) for m in rr.modifications]
    patch = workdir / "repair.diff"
    patch.write_text(rr.diff, encoding="utf-8")
    approved = jv.verdict == "ACCEPT" and not rr.requires_human_approval
    rationale = jv.reason or "; ".join(c.rationale for c in jv.criteria if c.result != "PASS")[:500]
    if rr.requires_human_approval:
        rationale = ("réparation à approuver par un humain (" + ", ".join(sorted({f.kind for f in rr.scope_flags}))
                     + ") ; " + rationale)[:500]
    result = RepairResult(attempt=attempt, source_report=inp.report_path, repaired_project=rr.workspace_path,
                          patch_path=str(patch), changes=changes,
                          judge=ContractVerdict(approved=approved, rationale=rationale or jv.verdict))
    outputs["repair_result"] = write_output(inp, "repair_result", result)
    outputs["repaired_project"] = rr.workspace_path
    warnings = [f"{summary} ; réparation {rr.status} ({len(changes)} modification(s)) ; juge : {jv.verdict}"]
    if not approved:
        return _fail(started, [f"réparation non retenue, juge : {jv.verdict} ({rationale[:250]})"], outputs, warnings)
    return AgentResult(agent="repair", status=AgentStatus.OK, can_proceed=True, outputs=outputs, warnings=warnings,
                       duration_s=round(time.monotonic() - started, 3))
