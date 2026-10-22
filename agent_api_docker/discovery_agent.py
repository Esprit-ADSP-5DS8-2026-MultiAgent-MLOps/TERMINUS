"""Agent de localisation : le LLM explore le depot pour retrouver le rapport d'analyse et le code du projet.

Repartition des roles (meme principe que pour le push Docker) :
  - le LLM decide OU chercher et QUEL candidat choisir ;
  - le CODE verifie ses choix : sandbox en lecture seule, contrat du rapport (pydantic), coherence du
    projet, presence du fichier d'inference. Un choix invalide est renvoye au LLM avec la raison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from . import config
from .llm import LLMClient, make_client
from .report_loader import ReportError
from .schemas import AnalysisReport
from .tools.discovery import Sandbox, SandboxError, classify_report, norm

MAX_TOOL_OUTPUT = 4000


class DiscoveryFailed(ReportError):
    """L'agent n'a pas trouve de rapport valide (ou demande de l'aide a l'utilisateur)."""


@dataclass
class DiscoveryResult:
    report_path: Path
    project_dir: Optional[Path]
    steps: int
    events: list[dict] = field(default_factory=list)  # trajectoire complete, pour les traces


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("list_dir", "Liste le contenu d'un dossier (chemin relatif a la racine du depot).",
          {"path": {"type": "string", "description": "ex: '.', 'eval', 'eval/projects/credit_risk'"}}, ["path"]),
    _tool("find_files", "Cherche des fichiers par motif glob sur le NOM (ex: '*analysis_report*.json').",
          {"pattern": {"type": "string"}, "root": {"type": "string", "description": "dossier de depart, defaut '.'"}},
          ["pattern"]),
    _tool("inspect_json", "Dit si un JSON est un rapport d'analyse, un rapport de structure ou autre, et pour quel projet.",
          {"path": {"type": "string"}}, ["path"]),
    _tool("finish", "Valide ton choix. Le code revalide le rapport et le dossier du projet.",
          {"report_path": {"type": "string"},
           "project_dir": {"type": "string", "description": "dossier du code source (facultatif si evident)"}},
          ["report_path"]),
    _tool("give_up", "Abandonne avec la raison et la question a poser a l'utilisateur.",
          {"reason": {"type": "string"}}, ["reason"]),
]


def _load_prompt(project_id: str, max_steps: int) -> str:
    text = (config.PROMPTS_DIR / "find_report.md").read_text(encoding="utf-8")
    return text.replace("{project_id}", project_id).replace("{max_steps}", str(max_steps))


def locate_project_dir(report: AnalysisReport, report_path: Path, project_id: str, sandbox: Sandbox,
                       projects_dir: Optional[Path] = None) -> Optional[Path]:
    """Dossier du code = celui qui contient le fichier d'inference du rapport. Deterministe."""
    entry = report.inference.entrypoint.file
    candidates: list[Path] = []
    pdir = Path(projects_dir) if projects_dir else config.EVAL_PROJECTS_DIR
    if pdir.is_dir():
        candidates += [d for d in sorted(pdir.iterdir()) if d.is_dir() and norm(d.name) == norm(project_id)]
    d = report_path.parent
    while True:                                  # le rapport est souvent range DANS le projet (ex: expected_reports/)
        candidates.append(d)
        if d == sandbox.base or d.parent == d:
            break
        d = d.parent
    for c in candidates:
        if (c / entry).is_file():
            return c.resolve()
    return None


def _check_choice(project_id: str, report_path: str, project_dir: Optional[str], sandbox: Sandbox,
                  projects_dir: Optional[Path]) -> tuple[Optional[tuple[Path, Optional[Path]]], str]:
    """Retourne ((rapport, dossier), message). Premier element None = choix refuse, message = raison."""
    f = sandbox.resolve(report_path)
    if not f.is_file():
        return None, f"Fichier introuvable : {report_path}"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        report = AnalysisReport.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"JSON illisible : {e}"
    except ValidationError:
        kind, detail = classify_report(data if isinstance(data, dict) else {})
        return None, f"REFUSE - ce n'est pas un rapport d'analyse valide ({kind}). {detail}"

    if norm(report.project_name) != norm(project_id) and norm(project_id) not in norm(f.as_posix()):
        return None, (f"REFUSE - ce rapport concerne le projet '{report.project_name}', pas '{project_id}'. "
                      "Cherche le rapport du bon projet.")

    if project_dir:
        d = sandbox.resolve(project_dir)
        if not (d / report.inference.entrypoint.file).is_file():
            return None, (f"REFUSE - {project_dir} ne contient pas le fichier d'inference "
                          f"'{report.inference.entrypoint.file}' attendu par le rapport.")
    else:
        d = locate_project_dir(report, f, project_id, sandbox, projects_dir)
        if d is None:
            return None, (f"Rapport valide, mais aucun dossier contenant '{report.inference.entrypoint.file}' "
                          "n'a ete trouve automatiquement. Cherche ce fichier avec find_files puis rappelle "
                          "finish avec project_dir.")
    return (f.resolve(), d), "ok"


def find_report(project_id: str, llm: Optional[LLMClient] = None, sandbox: Optional[Sandbox] = None,
                projects_dir: Optional[Path] = None, max_steps: int = 8,
                on_event: Callable[[dict], None] = lambda e: None) -> DiscoveryResult:
    llm = llm or make_client()
    sandbox = sandbox or Sandbox()
    events: list[dict] = []

    def emit(kind: str, **kw) -> None:
        ev = {"kind": kind, **kw}
        events.append(ev)
        on_event(ev)

    messages: list[dict] = [
        {"role": "system", "content": _load_prompt(project_id, max_steps)},
        {"role": "user", "content": f"Trouve le rapport d'analyse et le dossier du projet `{project_id}`."},
    ]
    dispatch = {
        "list_dir": lambda a: sandbox.list_dir(a.get("path", ".")),
        "find_files": lambda a: sandbox.find_files(a["pattern"], a.get("root", ".")),
        "inspect_json": lambda a: sandbox.inspect_json(a["path"]),
    }

    nudges = 0
    for step in range(1, max_steps + 1):
        resp = llm.chat(messages, TOOLS)
        if resp.text.strip():
            emit("thought", step=step, text=resp.text.strip())
        assistant: dict = {"role": "assistant", "content": resp.text or None}
        if resp.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
                                        **({"extra_content": c.extra} if c.extra else {})}   # thought_signature Gemini
                                       for c in resp.tool_calls]
        messages.append(assistant)

        if not resp.tool_calls:  # le LLM bavarde : on le ramene vers les outils
            nudges += 1
            if nudges > 2:
                raise DiscoveryFailed("L'agent n'utilise pas les outils (reponse en texte libre) : abandon.")
            messages.append({"role": "user", "content": "Utilise un outil (list_dir, find_files, inspect_json, finish ou give_up)."})
            continue

        for call in resp.tool_calls:
            args = call.arguments
            emit("tool", step=step, name=call.name, args=args)
            if "_invalid_json" in args:
                out = "ERREUR : arguments JSON invalides, reessaie."
            elif call.name == "give_up":
                emit("give_up", step=step, reason=args.get("reason", ""))
                raise DiscoveryFailed(args.get("reason") or "Aucun rapport trouve.")
            elif call.name == "finish":
                try:
                    choice, msg = _check_choice(project_id, args.get("report_path", ""), args.get("project_dir"),
                                                sandbox, projects_dir)
                except SandboxError as e:
                    choice, msg = None, f"ERREUR : {e}"
                if choice:
                    emit("done", step=step, report=sandbox.rel(choice[0]),
                         project_dir=sandbox.rel(choice[1]) if choice[1] else None)
                    return DiscoveryResult(choice[0], choice[1], step, events)
                out = msg
            elif call.name in dispatch:
                try:
                    out = dispatch[call.name](args)
                except SandboxError as e:
                    out = f"ERREUR : {e}"
                except (KeyError, TypeError):
                    out = f"ERREUR : arguments invalides pour {call.name} : {args}"
            else:
                out = f"ERREUR : outil inconnu '{call.name}'."
            out = out if len(out) <= MAX_TOOL_OUTPUT else out[:MAX_TOOL_OUTPUT] + "\n... (tronque)"
            emit("result", step=step, name=call.name, text=out)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})

    raise DiscoveryFailed(f"Budget de {max_steps} etapes epuise sans rapport valide.")
