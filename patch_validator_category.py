"""Patch idempotent de validator_agent/tools.py (lancer depuis la racine du projet : python patch_validator_category.py).
Problème : dans l'outil `submit_report`, le champ `issues[].category` n'avait pas de liste de valeurs permises ; le modèle y mettait
des valeurs de `nature` (ex. "configuration") -> rapport refusé (INVALID_SUBMISSION). Correctif : exposer la liste exacte des catégories
(dérivée de `agentic_core.schemas.Category`, donc toujours synchronisée). Aucune logique métier ne change."""
import pathlib, sys

p = pathlib.Path("validator_agent/tools.py")
if not p.exists(): sys.exit("lance ce script depuis la racine du projet")
s = p.read_text(encoding="utf-8")
done = []

OLD = '"category": {"type": "string"}, "severity"'
NEW = '"category": {"type": "string", "enum": list(get_args(Category))}, "severity"'
if "get_args(Category)" not in s:
    assert s.count(OLD) == 1, "ancre introuvable ou ambiguë (tools.py a changé ?)"
    s = s.replace(OLD, NEW, 1); done.append("enum category")

if "from typing import get_args" not in s:
    s = s.replace("import tomllib\n", "import tomllib\nfrom typing import get_args\n", 1); done.append("import get_args")
if "import Category" not in s and "Category," not in s.split("class ")[0]:
    s = s.replace("from agentic_core.sandbox_exec import COMMANDS, SandboxRunner\n",
                  "from agentic_core.sandbox_exec import COMMANDS, SandboxRunner\nfrom agentic_core.schemas import Category\n", 1); done.append("import Category")

p.write_text(s, encoding="utf-8")
print("appliqué :", ", ".join(done) if done else "rien (déjà à jour)")