from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .contract import NEVER_WAIVED, ContractConfig
from .models import Finding, Manifest, Severity


@dataclass
class Context:
    root: Path
    cfg: ContractConfig
    files: list[Path] = field(default_factory=list)      # chemins relatifs
    symlinks: list[Path] = field(default_factory=list)
    manifest: Optional[Manifest] = None
    manifest_state: str = "unknown"                       # ok | missing | invalid
    facts: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    observed_metrics: Optional[dict] = None
    sandbox_info: Optional[dict] = None
    level: str = ""

    def add(self, rule, code, severity, message, *, file=None, line=None, hint=None, evidence=None):
        if severity == Severity.BLOCKER and code in self.cfg.waived_codes and code not in NEVER_WAIVED:
            # un humain a donné raison au validateur v2 : le problème reste visible, il ne bloque plus
            severity = Severity.WARNING
            message = f"{message} [bloquant levé par {self.cfg.waived_by or 'un humain'}]"
        self.findings.append(Finding(
            rule=rule, code=code, severity=severity, level=self.level, message=message,
            file=str(file) if file is not None else None, line=line, hint=hint, evidence=evidence,
        ))

    def has(self, severity: Severity) -> bool:
        return any(f.severity == severity for f in self.findings)

    def py_files(self):
        return [p for p in self.files if p.suffix == ".py"]
