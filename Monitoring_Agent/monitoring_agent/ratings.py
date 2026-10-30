"""
Avis des utilisateurs (pouce haut / bas) sur les décisions de l'agent et sur les réponses de l'assistant.

Pourquoi : c'est la donnée qui manque pour améliorer le modèle fine-tuné avec de vrais cas de production.
- décision approuvée (haut)        -> exemple d'entraînement tel quel ;
- décision corrigée (bas + bonne cause / bonne action) -> exemple d'entraînement avec la correction ;
- décision refusée sans correction  -> gardée comme exemple « rejeté » (utile pour une méthode de
  préférences type DPO plus tard), mais pas comme exemple d'entraînement.

Tout est écrit en JSONL dans le dossier du projet (ratings.jsonl), une ligne par avis.
`python -m monitoring_agent ratings <projet> --export fichier.jsonl` sort les exemples au format MLX.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .agent.schemas import ACTIONS, CAUSES

FILE = "ratings.jsonl"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def decision_example(decision: dict, messages: list[dict] | None, rating: str,
                     correct_cause: str | None, correct_action: str | None) -> dict | None:
    """Exemple au format chat {system, user, assistant}, ou None si on ne peut pas en faire un."""
    diag = decision.get("diagnosis")
    if not messages or not diag:
        return None
    if rating == "down":
        if not (correct_cause or correct_action):
            return None
        diag = {**diag, "root_cause": correct_cause or diag["root_cause"],
                "action": correct_action or diag["action"],
                "rationale": f"Corrigé par un humain. {diag.get('rationale', '')}".strip()}
    keep = ("root_cause", "summary", "affected_features", "evidence", "action", "rationale", "confidence")
    target = {k: diag[k] for k in keep if k in diag}
    return {"messages": [*messages, {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]}


def add(project_dir: Path, kind: str, ref: str, rating: str, comment: str = "", snapshot: dict | None = None,
        example: dict | None = None, correct_cause: str | None = None, correct_action: str | None = None,
        by: str = "console") -> dict:
    if kind not in ("decision", "answer"):
        raise ValueError("kind : decision ou answer")
    if rating not in ("up", "down"):
        raise ValueError("rating : up ou down")
    if correct_cause and correct_cause not in CAUSES:
        raise ValueError(f"cause inconnue : {correct_cause}")
    if correct_action and correct_action not in ACTIONS:
        raise ValueError(f"action inconnue : {correct_action}")
    rec = {"id": f"R{uuid.uuid4().hex[:8]}", "kind": kind, "ref": ref, "rating": rating,
           "comment": comment.strip()[:2000], "correct_cause": correct_cause, "correct_action": correct_action,
           "by": by, "created_at": datetime.now(UTC).isoformat(), "snapshot": snapshot or {}, "example": example}
    project_dir.mkdir(parents=True, exist_ok=True)
    with (project_dir / FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def latest(project_dir: Path) -> dict[str, dict]:
    """Dernier avis pour chaque élément (on peut changer d'avis : le plus récent compte)."""
    out: dict[str, dict] = {}
    for r in _read(project_dir / FILE):
        out[f"{r['kind']}:{r['ref']}"] = r
    return out


def summary(project_dir: Path) -> dict:
    items = list(latest(project_dir).values())
    return {"total": len(items),
            "decisions": {"up": sum(r["kind"] == "decision" and r["rating"] == "up" for r in items),
                          "down": sum(r["kind"] == "decision" and r["rating"] == "down" for r in items)},
            "answers": {"up": sum(r["kind"] == "answer" and r["rating"] == "up" for r in items),
                        "down": sum(r["kind"] == "answer" and r["rating"] == "down" for r in items)},
            "training_examples": sum(bool(r.get("example")) for r in items),
            "by_ref": {k: r["rating"] for k, r in latest(project_dir).items()}}


def export(project_dir: Path, out: Path) -> int:
    """Exemples d'entraînement (dernier avis de chaque décision), au même format que finetune/build.py."""
    examples = [r["example"] for r in latest(project_dir).values() if r.get("example")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
    return len(examples)


def answer_snapshot(project_dir: Path, answer_id: str) -> dict | None:
    """La question et la réponse du chat notées (assistant.jsonl)."""
    path = project_dir / "assistant.jsonl"
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("id") == answer_id:
            return {k: rec.get(k) for k in ("question", "answer", "tools", "model", "window", "prompt")}
    return None


def diagnoser_messages(supervisor, decision_id: str) -> list[dict] | None:
    """Le prompt du diagnostiqueur pour cette décision, au format du fine-tuning (faits seuls)."""
    if supervisor is None or decision_id not in supervisor.agent.cases:
        return None     # gateway redémarrée : le dossier n'est plus en mémoire
    from .agent import prompts
    from .agent.context import case_facts
    from .agent.graph import build_messages
    case = supervisor.agent.cases[decision_id]
    return build_messages(prompts.DIAGNOSER, case, case_facts(case))


def rate(project_dir: Path, store, supervisor, kind: str, ref: str, rating: str, comment: str = "",
         correct_cause: str | None = None, correct_action: str | None = None, by: str = "console") -> dict:
    """Enregistre un avis venant de la console ou de Slack. Lève KeyError si l'élément noté n'existe pas."""
    snapshot, example = {}, None
    if kind == "decision":
        d = store.decision(ref)
        if d is None:
            raise KeyError("décision inconnue")
        snapshot = {k: d.get(k) for k in ("action", "status", "mode", "window_index", "signals")}
        snapshot["diagnosis"] = d.get("diagnosis")
        example = decision_example(d, diagnoser_messages(supervisor, ref), rating, correct_cause, correct_action)
    elif kind == "answer":
        snapshot = answer_snapshot(project_dir, ref) or {}
    rec = add(project_dir, kind, ref, rating, comment, snapshot, example, correct_cause, correct_action, by)
    return {"id": rec["id"], "training_example": bool(example), **summary(project_dir)}
