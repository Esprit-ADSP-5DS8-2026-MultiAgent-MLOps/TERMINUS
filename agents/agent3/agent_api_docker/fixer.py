"""Correction automatique : quand la validation ou les tests echouent, le LLM propose des patchs.

Repartition des roles (meme principe que la recherche du rapport et le push Docker) :
  - le LLM diagnostique et propose des patchs {path, old, new} ;
  - le CODE controle tout : seuls les fichiers GENERES sont modifiables (jamais le code du projet, jamais
    les tests, qui sont regeneres depuis le rapport), `old` doit apparaitre exactement une fois, une
    correction deja essayee est refusee, le nombre de tentatives est plafonne.
Chaque tentative est ecrite dans traces.jsonl (step "fix") : c'est le dataset du futur fine-tuning.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .errors import classify, tail_relevant
from .llm import LLMClient, LLMError
from .schemas import AnalysisReport
from .tools.files import patch_file

# Seuls ces fichiers (generes) peuvent etre corriges par le LLM.
FIXABLE_FILES = ("api.py", "requirements.api.txt", "Dockerfile", ".dockerignore", "docker-compose.yml")
MAX_FILE_CHARS = 8000
MAX_PATCHES = 6


class FixError(LLMError):
    """Le LLM n'a pas donne de correction exploitable."""


@dataclass
class Patch:
    path: str
    old: str
    new: str

    def key(self) -> str:
        return hashlib.sha1(f"{self.path}\0{self.old}\0{self.new}".encode()).hexdigest()


@dataclass
class FixProposal:
    diagnosis: str
    target: str
    patches: list[Patch]
    model: str = ""


@dataclass
class FixAttempt:
    """Une tentative de correction (pour l'historique donne au LLM et pour les traces)."""
    iteration: int
    category: str
    failure: str
    diagnosis: str = ""
    target: str = ""
    patches: list[dict] = field(default_factory=list)
    applied: int = 0
    errors: list[str] = field(default_factory=list)
    model: str = ""
    stage: str = ""      # "" = validation/tests locaux ; "build" | "run" | "health" | "tests" | "lint" = phase Docker

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- contexte donne au LLM

def _read(path: Path, limit: int = MAX_FILE_CHARS) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n... (tronque)"


def _report_excerpt(report: AnalysisReport) -> str:
    keep = {
        "python": report.python.model_dump(),
        "dependencies": report.dependencies.model_dump(),
        "ml": report.ml.model_dump(),
        "inference": report.inference.model_dump(),
        "io": report.io.model_dump(),
        "existing_app": report.existing_app.model_dump(),
    }
    return json.dumps(keep, ensure_ascii=False, indent=1, default=str)[:6000]


def build_context(report: AnalysisReport, generated_dir: Path, run_dir: Path, category: str, failure: str,
                  history: list[FixAttempt], context_files: Optional[list[str]] = None) -> str:
    parts = [f"## Categorie d'erreur : {category}", "## Echec a corriger\n" + failure]
    if context_files:     # phase Docker : ce que le build voit reellement (utile pour les erreurs COPY)
        parts.append("## Fichiers du contexte de build Docker (racine = dossier du Dockerfile)\n" + "\n".join(context_files))
    for name in FIXABLE_FILES:
        f = Path(generated_dir) / name
        if f.is_file():
            parts.append(f"## Fichier courant : {name}\n```\n{_read(f)}\n```")
    entry = Path(run_dir) / report.inference.entrypoint.file
    if entry.is_file():                       # en lecture seule : sert a comprendre la signature des fonctions
        parts.append(f"## Code d'inference du projet (lecture seule, NON modifiable) : "
                     f"{report.inference.entrypoint.file}\n```\n{_read(entry, 5000)}\n```")
    parts.append("## Rapport d'analyse (extrait)\n" + _report_excerpt(report))
    if history:
        lines = []
        for h in history:
            outcome = "appliquee" if h.applied else "NON appliquee"
            lines.append(f"- tentative {h.iteration} ({outcome}) : {h.diagnosis}\n"
                         f"  patchs : {json.dumps(h.patches, ensure_ascii=False)[:1200]}\n"
                         f"  erreurs d'application : {h.errors or 'aucune'}\n"
                         f"  echec qui a suivi : {h.failure[:500]}")
        parts.append("## Corrections DEJA ESSAYEES (ne pas les repeter)\n" + "\n".join(lines))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- appel du LLM

def _extract_json(text: str) -> dict:
    """Accepte du JSON nu, entoure de ```json ... ```, ou noye dans du texte."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError("pas de JSON valide dans la reponse")


def _parse(data: dict) -> tuple[str, str, list[Patch]]:
    patches = []
    for p in data.get("patches") or []:
        if not isinstance(p, dict) or not {"path", "old", "new"} <= set(p):
            raise ValueError(f"patch mal forme (attendu path/old/new) : {str(p)[:200]}")
        patches.append(Patch(str(p["path"]), str(p["old"]), str(p["new"])))
    return str(data.get("diagnosis", "")), str(data.get("target", "")), patches


def propose_fix(llm: LLMClient, context: str, max_format_retries: int = 2) -> FixProposal:
    """Demande une correction au LLM (JSON uniquement). Redemande si le format est invalide."""
    system = (config.PROMPTS_DIR / "fix_error.md").read_text(encoding="utf-8")
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": context}]
    last = ""
    for _ in range(max_format_retries + 1):
        resp = llm.chat(messages, [])
        try:
            diagnosis, target, patches = _parse(_extract_json(resp.text))
            return FixProposal(diagnosis, target, patches, resp.model)
        except ValueError as e:
            last = f"{e} | reponse : {resp.text[:200]!r}"
            messages.append({"role": "assistant", "content": resp.text or ""})
            messages.append({"role": "user", "content": f"Format invalide ({e}). Reponds UNIQUEMENT avec l'objet JSON "
                                                        '{"diagnosis": ..., "target": ..., "patches": [{"path","old","new"}]}.'})
    raise FixError(f"Le LLM n'a pas renvoye de JSON exploitable : {last}")


# --------------------------------------------------------------------------- application (controlee par le code)

def apply_patches(generated_dir: Path, patches: list[Patch], already_tried: set[str],
                  applied_out: Optional[list[Patch]] = None) -> tuple[int, list[str]]:
    """Applique les patchs valides. Retourne (nombre applique, erreurs expliquees au LLM).

    applied_out : si fourni, y ajoute chaque patch effectivement applique (pour les memoriser, voir fix_store.py).
    """
    applied, errors = 0, []
    if len(patches) > MAX_PATCHES:
        errors.append(f"trop de patchs ({len(patches)}), seuls les {MAX_PATCHES} premiers sont consideres")
        patches = patches[:MAX_PATCHES]
    for p in patches:
        if p.path not in FIXABLE_FILES:
            errors.append(f"{p.path} : fichier non modifiable (autorises : {', '.join(FIXABLE_FILES)})")
        elif not p.old:
            errors.append(f"{p.path} : 'old' vide, impossible de localiser la zone a remplacer")
        elif p.old == p.new:
            errors.append(f"{p.path} : 'old' et 'new' identiques")
        elif p.key() in already_tried:
            errors.append(f"{p.path} : ce patch a deja ete essaye, propose une autre correction")
        elif not (Path(generated_dir) / p.path).is_file():
            errors.append(f"{p.path} : fichier absent des fichiers generes")
        else:
            try:
                patch_file(Path(generated_dir), p.path, p.old, p.new)
            except ValueError as e:
                errors.append(str(e))
            else:
                applied += 1
                if applied_out is not None:
                    applied_out.append(p)
        already_tried.add(p.key())
    return applied, errors


def classify_failure(failure: str, hint: Optional[str] = None) -> str:
    """Categorie deja calculee par la validation/les tests, sinon deduite des logs."""
    return hint or classify(failure)[0]


def short_failure(text: str, max_lines: int = 60) -> str:
    return tail_relevant(text, max_lines)[-4000:]
