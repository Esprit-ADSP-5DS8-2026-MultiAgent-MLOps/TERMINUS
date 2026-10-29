"""État partagé du workflow (minimal : à étendre à l'implémentation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkflowState:
    project: str                                   # projet SOURCE (jamais modifié)
    workspace: Optional[str] = None                # copie corrigée produite par le Repair
    attempt: int = 0
    max_attempts: int = 2
    reports: list = field(default_factory=list)    # rapports dans l'ordre du lignage
    final_admission: Optional[str] = None          # lu UNIQUEMENT sur un ValidatorReport, jamais sur le Judge
