from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

class CritSub(BaseModel): ...      # copie exacte depuis evaluator.py
class VerdictSub(BaseModel): ...   # copie exacte depuis evaluator.py

class CritSub(BaseModel):
    name: Literal["correctness", "completeness", "regression_risk", "evidence_quality", "scope_compliance"]
    result: Literal["PASS", "CONCERN", "FAIL"]
    rationale: str
    evidence: Optional[str] = None


class VerdictSub(BaseModel):
    verdict: Literal["ACCEPT", "REVISE", "REJECT"]
    criteria: list[CritSub]
    evidence_examined: list[str] = Field(default_factory=list)
    remaining_concerns: list[str] = Field(default_factory=list)
    reason: str = ""
    next_action: Literal["REVALIDATE", "RETRY", "ESCALATE", "ROLLBACK"]

