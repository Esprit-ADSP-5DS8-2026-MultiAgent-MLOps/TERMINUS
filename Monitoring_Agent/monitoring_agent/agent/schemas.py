"""
Ce que l'agent PRODUIT : diagnostic, verdict du juge, décision de la policy, trace complète.

Les modèles lus depuis un LLM sont tolérants (extra="ignore") : un champ en trop ne fait pas
échouer la validation, mais un champ manquant ou une valeur hors liste, si.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RootCause = Literal["covariate_shift", "concept_drift", "data_quality", "service_degradation",
                    "benign_drift", "no_issue"]
Action = Literal["no_action", "increase_monitoring", "open_incident", "notify_data_team",
                 "request_retraining", "rollback_model"]

# Risque d'une action = ce qu'elle coûte si elle est fausse, et si on peut l'annuler.
ACTION_RISK: dict[str, str] = {
    "no_action": "low",            # rien ne change
    "increase_monitoring": "low",  # on regarde de plus près
    "open_incident": "low",        # un rapport écrit
    "notify_data_team": "low",     # un rapport à l'équipe qui produit les données
    "request_retraining": "medium",  # une DEMANDE : orchestrateur + quality gate de l'Agent 2 derrière
    "rollback_model": "high",      # change le modèle servi : toujours un humain
}


JUDGE_CRITERIA = ("grounding", "causality", "proportionality", "safety")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Evidence(LLMModel):
    fact: str = Field(description="clé exacte d'un fait de <faits>")
    value: float = Field(description="valeur du fait, recopiée")
    claim: str = Field(description="ce que ce fait montre, en une phrase")


class Diagnosis(LLMModel):
    root_cause: RootCause
    summary: str
    affected_features: list[str] = []
    evidence: list[Evidence] = []
    action: Action
    rationale: str
    confidence: float = Field(ge=0, le=1)


class JudgeVerdict(LLMModel):
    verdict: Literal["approve", "revise", "reject"]
    # dict côté code (tolérant), mais schéma explicite côté LLM : les 4 critères, notes entières de 1 à 5.
    scores: dict[str, int] = Field(default={}, json_schema_extra={
        "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in JUDGE_CRITERIA},
        "required": list(JUDGE_CRITERIA)})
    critique: str = ""


class EvidenceCheck(BaseModel):
    fact: str
    claimed: float
    actual: float | None
    ok: bool


class Verification(BaseModel):
    ok: bool
    checks: list[EvidenceCheck] = []
    problems: list[str] = []


class RuleCheck(BaseModel):
    rule: str
    passed: bool
    detail: str


class PolicyDecision(BaseModel):
    proposed_action: Action
    final_action: Action
    mode: Literal["auto", "human", "blocked"]   # auto : l'agent agit seul ; human : il attend ; blocked : refusé
    rules: list[RuleCheck] = []
    reasons: list[str] = []


class TraceStep(BaseModel):
    node: str
    ms: float
    summary: str
    detail: dict = {}


class AgentRun(BaseModel):
    """Tout ce que l'agent a fait pour UNE fenêtre : rejouable, auditable, affichable."""

    id: str
    project: str
    window_index: int
    created_at: datetime = Field(default_factory=utcnow)
    decider: str = ""                      # modèle (ou « rules ») qui a diagnostiqué
    judge: str = ""
    samples: list[Diagnosis] = []
    agreement: float = 0.0
    diagnosis: Diagnosis | None = None
    verification: Verification | None = None
    verdicts: list[JudgeVerdict] = []
    policy: PolicyDecision | None = None
    status: Literal["executed", "awaiting_human", "rejected_by_human", "blocked", "failed"] = "failed"
    outputs: dict[str, str] = {}
    human: dict | None = None
    trace: list[TraceStep] = []
    error: str | None = None

    @property
    def usage(self) -> dict:
        """Total des appels LLM de la décision (jetons, appels, temps passé dans les LLM)."""
        tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "ms": 0.0}
        for t in self.trace:
            u = t.detail.get("usage") or {}
            for k in tot:
                tot[k] += u.get(k, 0) or 0
        return tot
