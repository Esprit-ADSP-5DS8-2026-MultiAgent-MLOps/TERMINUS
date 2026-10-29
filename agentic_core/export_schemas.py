"""Exporte le schéma JSON des trois rapports : contrat d'interface pour l'orchestrateur (LangGraph…).
    python -m agentic_core.export_schemas docs/schemas
"""
import sys
from pathlib import Path

from .schemas import MODELS, schema_json


def export(dest) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name, model in MODELS.items():
        p = dest / f"{name}.schema.json"
        p.write_text(schema_json(model) + "\n", encoding="utf-8")
        out.append(p)
    return out


if __name__ == "__main__":
    for p in export(sys.argv[1] if len(sys.argv) > 1 else "docs/schemas"):
        print(p)
