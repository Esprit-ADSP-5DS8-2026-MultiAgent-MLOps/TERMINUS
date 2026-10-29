#!/usr/bin/env bash
# Réapplicable : part de l'état ACTUEL du projet, refait ce qui manque, corrige ce qui est cassé. Sans git.
# Usage : depuis la racine du projet :  bash apply_refactor.sh
set -uo pipefail

if [ ! -f pyproject.toml ] || [ ! -d repair_agent ]; then echo "ERREUR : lance le script depuis la racine du projet (pyproject.toml + repair_agent/)"; exit 1; fi

echo "[0] Sauvegarde"
BK="../$(basename "$PWD")_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$BK" --exclude=.venv --exclude=venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=out -C .. "$(basename "$PWD")" \
  && echo "  $BK   (restauration : tar xzf $BK -C ..)" || { echo "ERREUR sauvegarde : arrêt"; exit 1; }
export BK

python3 - <<'PY'
import pathlib, re, shutil, sys, os

ROOT = pathlib.Path(".")
LEG, J = ROOT / "judge_agent", ROOT / "repair_agent" / "judge"
warns = []
def info(m): print("   ", m)
def warn(m): print("    AVERTISSEMENT :", m); warns.append(m)
def read(p): return p.read_text(encoding="utf-8")
def write(p, s): p.write_text(s, encoding="utf-8")
def skip(p): return any(x in p.parts for x in (".venv", "venv", "__pycache__", "site-packages", ".git")) or "eval/projects" in p.as_posix() or "egg-info" in p.as_posix()

# ---- pré-vérification : où est le code de l'évaluateur ? (avant toute modification)
if not any(p.exists() for p in (J / "evaluator.py", J / "agent.py", LEG / "agent.py")):
    print("ERREUR : aucun evaluator.py / agent.py trouvé. Contenu des dossiers concernés :")
    for d in (LEG, J):
        print(f"  {d}/ :", sorted(x.name for x in d.iterdir()) if d.is_dir() else "(absent)")
    print("Je m'arrête sans rien modifier. Envoie-moi cette liste.")
    sys.exit(2)

print("[1] Déplacement / normalisation de l'arborescence du Judge")
J.mkdir(parents=True, exist_ok=True)
if LEG.is_dir():
    for f in sorted(LEG.glob("*.py")):
        if f.name == "__init__.py": continue
        dest = J / ("evaluator.py" if f.name == "agent.py" else f.name)
        if dest.exists(): warn(f"{dest} existe déjà : conservé ; {f} laissé en place (compare à la main, voir sauvegarde)")
        else: shutil.move(str(f), str(dest)); info(f"déplacé {f} -> {dest}")
    sysmd = LEG / "prompts" / "system.md"
    if sysmd.exists():
        if (J / "system.md").exists(): warn(f"{J/'system.md'} existe déjà : {sysmd} laissé en place")
        else: shutil.move(str(sysmd), str(J / "system.md")); info("déplacé prompts/system.md -> judge/system.md")
    (LEG / "__init__.py").unlink(missing_ok=True)
    for d in (LEG / "__pycache__", LEG / "prompts" / "__pycache__"): shutil.rmtree(d, ignore_errors=True)
    for d in (LEG / "prompts", LEG):
        try: d.rmdir()
        except OSError: pass
    if LEG.exists(): warn("judge_agent/ n'est pas vide, laissé en place : vérifie son contenu puis supprime-le")
    else: info("judge_agent/ supprimé")
# prompts/ dans judge/ (entrerait en conflit avec prompts.py)
pd = J / "prompts"
if pd.is_dir() and not (pd / "__init__.py").exists():
    if (pd / "system.md").exists() and not (J / "system.md").exists(): shutil.move(str(pd / "system.md"), str(J / "system.md")); info("judge/prompts/system.md -> judge/system.md")
    shutil.rmtree(pd / "__pycache__", ignore_errors=True)
    try: pd.rmdir()
    except OSError: warn("judge/prompts/ non vide, à vérifier")
# agent.py résiduel dans judge/
if (J / "agent.py").exists():
    if not (J / "evaluator.py").exists(): (J / "agent.py").rename(J / "evaluator.py"); info("judge/agent.py -> judge/evaluator.py")
    else: warn("judge/agent.py ET judge/evaluator.py existent : evaluator.py conservé, agent.py à supprimer à la main")
# tools.py introuvable ? chercher la classe ailleurs
if not (J / "tools.py").exists():
    found = [p for p in ROOT.rglob("*.py") if not skip(p) and J not in p.parents and "class JudgeToolBox" in read(p)]
    if found: shutil.copy2(found[0], J / "tools.py"); info(f"tools.py récupéré depuis {found[0]} (copie)")
    else: warn("tools.py introuvable : evaluator.py l'importe (JudgeToolBox). Récupère-le depuis ton zip d'origine : judge_agent/tools.py")

print("[2] Découpage evaluator.py -> schemas.py / prompts.py / __init__.py")
ev = J / "evaluator.py"; s = read(ev)
blk = re.search(r"class CritSub\(BaseModel\):.*?(?=^def _checker\()", s, re.S | re.M)
if blk:
    if not (J / "schemas.py").exists():
        write(J / "schemas.py",
              '"""Schémas de soumission du Judge (ce que le LLM dépose via `submit_verdict`).\n\n'
              "Le verdict final scellé (`JudgeVerdict`, `Criterion`) reste dans `agentic_core.schemas` (contrat partagé).\n\"\"\"\n"
              "from __future__ import annotations\n\nfrom typing import Literal, Optional\n\nfrom pydantic import BaseModel, Field\n\n\n" + blk.group(0).rstrip() + "\n")
        info("schemas.py créé")
    s = s.replace(blk.group(0), "")
    info("CritSub/VerdictSub retirés de evaluator.py")
elif not (J / "schemas.py").exists():
    warn("ni CritSub dans evaluator.py ni schemas.py : à vérifier")
write(J / "prompts.py", '"""Prompt système du Judge (texte dans `system.md`)."""\nfrom pathlib import Path\n\nPROMPT = (Path(__file__).parent / "system.md").read_text(encoding="utf-8")\n')
if not (J / "system.md").exists(): warn("judge/system.md introuvable (prompts.py ne pourra pas le lire)")
s = re.sub(r"^PROMPT\s*=.*\n", "", s, flags=re.M)
def ensure_import(s, line):
    if line in s: return s
    m = re.search(r"^from \.tools import .*\n", s, re.M)
    if m: return s[:m.start()] + line + "\n" + s[m.start():]
    m = re.search(r"^ALLOWED_NEXT", s, re.M)
    return (s[:m.start()] + line + "\n\n" + s[m.start():]) if m else s + "\n" + line + "\n"
if "PROMPT" in s: s = ensure_import(s, "from .prompts import PROMPT")
if "VerdictSub" in s: s = ensure_import(s, "from .schemas import VerdictSub")
s = s.replace("from typing import Literal, Optional\n\nfrom pydantic import BaseModel, Field, ValidationError\n", "\nfrom pydantic import ValidationError\n")
s = re.sub(r"^from \.agent import .*\n", "", s, flags=re.M)
s = s.replace('"""Agent 3 — Judge. Fonction', '"""Judge (évaluateur intégré au Repair Agent). Fonction', 1)
s = s.replace("Jamais de décision d'admission. Échec fermé", "Jamais de décision d'admission (rôle exclusif du Validator, après revalidation). Échec fermé", 1)
write(ev, s)
write(J / "__init__.py", '"""Judge : composant du Repair Agent qui évalue la QUALITÉ d\'une réparation (lecture seule, preuves vérifiables).\n\n'
      "Il ne décide jamais de l'admission : après ACCEPT (next_action=REVALIDATE), le Validator revalide et décide.\n\"\"\"\n"
      "from .evaluator import JudgeConfig, run_judge  # noqa: F401\n")
mn = J / "__main__.py"
if mn.exists():
    t = read(mn)
    t = t.replace("python -m judge_agent PROJET", "python -m repair_agent.judge PROJET").replace('prog="judge_agent"', 'prog="repair_agent.judge"')
    t = re.sub(r"^from \.agent import", "from .evaluator import", t, flags=re.M)
    write(mn, t)
else: warn("judge/__main__.py absent (CLI python -m repair_agent.judge indisponible)")

print("[3] Imports, tests, eval")
SUBS = [(r"from judge_agent\.tools import", "from repair_agent.judge.tools import"),
        (r"from judge_agent\.agent import", "from repair_agent.judge.evaluator import"),
        (r"from judge_agent import", "from repair_agent.judge import"),
        (r"python -m judge_agent\b", "python -m repair_agent.judge")]
changed = 0
for p in ROOT.rglob("*.py"):
    if skip(p): continue
    t = read(p); o = t
    for a, b in SUBS: t = re.sub(a, b, t)
    if t != o: write(p, t); changed += 1; info(f"imports corrigés : {p}")
if not changed: info("aucun import à corriger")
old, new = ROOT / "tests" / "test_judge_agent.py", ROOT / "tests" / "test_judge.py"
if old.exists():
    if new.exists(): warn("tests/test_judge_agent.py ET tests/test_judge.py existent : à fusionner/supprimer à la main")
    else: old.rename(new); info("tests/test_judge_agent.py -> tests/test_judge.py")

print("[4] pyproject.toml")
pp = ROOT / "pyproject.toml"; t = read(pp)
m = re.search(r"packages\s*=\s*\[(.*?)\]", t, re.S)
if m:
    names = [n for n in re.findall(r'"([^"]+)"', m.group(1)) if n != "judge_agent"]
    for n in ("agentic_core.orchestrator", "repair_agent.judge"):
        if n not in names: names.append(n)
    t = t[:m.start()] + "packages = [" + ", ".join(f'"{n}"' for n in names) + "]" + t[m.end():]
else: info("pas de liste `packages` explicite (découverte automatique ?) : vérifie que repair_agent.judge et son .md sont inclus")
t = re.sub(r"^judge_agent\s*=.*\n", "", t, flags=re.M)
if '"repair_agent.judge"' not in t:
    if re.search(r"^repair_agent\s*=.*$", t, re.M): t = re.sub(r"^(repair_agent\s*=.*)$", r'\1\n"repair_agent.judge" = ["*.md"]', t, count=1, flags=re.M)
    elif "[tool.setuptools.package-data]" in t: t = t.replace("[tool.setuptools.package-data]", '[tool.setuptools.package-data]\n"repair_agent.judge" = ["*.md"]', 1)
    else: warn("section package-data absente : ajoute  \"repair_agent.judge\" = [\"*.md\"]  à la main")
write(pp, t)
if "judge_agent" in t: warn("'judge_agent' reste dans pyproject.toml")

print("[5] Squelette de l'orchestrateur (créé seulement s'il manque)")
O = ROOT / "agentic_core" / "orchestrator"; O.mkdir(parents=True, exist_ok=True)
FILES = {
"__init__.py": '''"""Orchestrateur (squelette, non implémenté) : Validator -> Repair (+ Judge intégré) -> revalidation.

Règle : le Judge évalue la qualité d'une réparation ; la décision finale d'admission appartient TOUJOURS au Validator,
après une nouvelle validation de la copie corrigée.
"""
from .state import WorkflowState  # noqa: F401
from .workflow import run_workflow  # noqa: F401
''',
"state.py": '''"""État partagé du workflow (minimal : à étendre à l'implémentation)."""
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
''',
"routing.py": '''"""Table de routage par `next_action` (contrat des schémas v2). Données seulement."""
ROUTES: dict[str, dict[str, str]] = {
    "validator": {"PROCEED": "END_ADMITTED", "REPAIR": "repair", "HUMAN_REVIEW": "END_HUMAN_REVIEW", "STOP": "END_REJECTED"},
    "repair": {"JUDGE": "judge", "ESCALATE": "END_HUMAN_REVIEW", "NONE": "END_NO_CHANGE"},
    "judge": {"REVALIDATE": "validator", "RETRY": "repair", "ESCALATE": "END_HUMAN_REVIEW", "ROLLBACK": "rollback"},
}
''',
"workflow.py": '''"""Point d'entrée du workflow global. NON IMPLÉMENTÉ volontairement."""
from __future__ import annotations

from .state import WorkflowState


def run_workflow(state: WorkflowState, client) -> WorkflowState:
    raise NotImplementedError("orchestrateur non implémenté : voir routing.py pour le contrat de routage")
'''}
for n, c in FILES.items():
    if (O / n).exists(): info(f"{n} déjà présent : conservé")
    else: write(O / n, c); info(f"{n} créé")

print("[6] .gitignore")
gi = ROOT / ".gitignore"; cur = read(gi).splitlines() if gi.exists() else []
add = [l for l in [".venv/", "venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".cache/", "out/", "*.egg-info/", "results_*.json", "build/", "dist/", "*.egg",
                   ".env", ".DS_Store", ".ruff_cache/", ".mypy_cache/", ".coverage", "htmlcov/", "tmp/", "*.tmp"] if l not in cur]
if add: write(gi, "\n".join(cur + add) + "\n"); info("ajouté : " + " ".join(add))
else: info("déjà complet")

print("[7] README")
rd = ROOT / "README.md"
if rd.exists():
    t = read(rd); o = t
    t = t.replace("python -m judge_agent $P", "python -m repair_agent.judge $P").replace("| `judge_agent/` |", "| `repair_agent/judge/` |")
    t = t.replace("judge_agent/  Agent 3", "(+ judge/ : évaluateur intégré)")
    t = re.sub(r"^(agentic_core/ +)(schémas v2)", r"\1orchestrator/ (squelette), \2", t, flags=re.M) if "orchestrator/" not in t else t
    if t != o: write(rd, t); info("README mis à jour")
    else: info("README déjà à jour")

print("[8] Contrôles")
bad = [p for p in J.glob("*.py") if compile(read(p), str(p), "exec") is None]
left = [f"{p}:{i+1}" for p in ROOT.rglob("*") if p.suffix in (".py", ".toml", ".md") and p.is_file() and not skip(p)
        for i, l in enumerate(read(p).splitlines()) if "judge_agent" in l]
print("    références restantes à judge_agent :", ", ".join(left) if left else "aucune")
for l in left: warns.append("référence restante " + l)
print("\n=== %d avertissement(s) ===" % len(warns))
for w in warns: print(" -", w)
PY
rc=$?
[ $rc -ne 0 ] && { echo "Le script s'est arrêté (code $rc). Sauvegarde : $BK"; exit $rc; }

find . \( -path ./.venv -o -path ./venv \) -prune -o -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo; echo "[9] Import + tests"
python3 -c "import repair_agent.judge, agentic_core.orchestrator; print('    imports OK')" || echo "    IMPORT EN ÉCHEC : voir les messages ci-dessus"
python3 -m pytest -q || echo "pytest a échoué ou n'est pas installé (sauvegarde : $BK)"
