"""Localisation, lecture et validation du rapport d'analyse."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from . import config
from .schemas import AnalysisReport


class ReportError(Exception):
    """Rapport introuvable, illisible ou invalide."""


class NeedsUserInput(Exception):
    """Le rapport demande une decision humaine avant de continuer."""

    def __init__(self, questions: list[str]):
        self.questions = questions
        super().__init__("Intervention utilisateur requise : " + " | ".join(questions))


def resolve_report_path(project_id: str, explicit: Optional[str | Path] = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("AGENT_REPORT_PATH"):
        candidates.append(Path(os.environ["AGENT_REPORT_PATH"]))
    candidates.append(config.WORKSPACES_DIR / project_id / config.REPORT_FILENAME)
    candidates.append(config.EVAL_REPORTS_DIR / f"{project_id}.analysis_report.json")
    # rapports ranges dans le dossier du projet lui-meme (ex: eval/projects/<id>/expected_reports/)
    project = config.EVAL_PROJECTS_DIR / project_id
    candidates.append(project / "expected_reports" / f"{project_id}.analysis_report.json")
    candidates.append(project / config.REPORT_FILENAME)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    tried = "\n  - ".join(str(c) for c in candidates)
    raise ReportError(f"Rapport introuvable. Chemins essayes :\n  - {tried}")


def load_report(project_id: str, explicit: Optional[str | Path] = None) -> AnalysisReport:
    path = resolve_report_path(project_id, explicit)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        report = AnalysisReport.model_validate(raw)
    except json.JSONDecodeError as e:
        raise ReportError(f"JSON invalide dans {path}: {e}") from e
    except ValidationError as e:
        raise ReportError(f"Rapport non conforme au contrat ({path}):\n{e}") from e

    questions = [u.question for u in report.unresolved]
    if report.needs_user_input or report.io.schema_confidence == "low":
        raise NeedsUserInput(questions or ["Schema d'entree incertain : precisez les features."])
    return report
