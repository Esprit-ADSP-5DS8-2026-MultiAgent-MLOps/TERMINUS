import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # agents/validator/
REPO = ROOT.parents[1]                              # racine du dépôt : validator/ (v1), common/
sys.path[:0] = [str(ROOT), str(REPO)]
os.environ["ORCH_AGENTS"] = "stub"          # tests du nœud repair dans le pipeline : les autres agents restent des stubs
os.environ["ORCH_SUPERVISOR"] = "rules"

if not (ROOT / "eval" / "projects" / "variants").exists():
    subprocess.run([sys.executable, str(ROOT / "eval" / "make_variants.py")], check=True)
