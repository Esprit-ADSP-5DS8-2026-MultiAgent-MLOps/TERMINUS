import os
import subprocess
import sys
from pathlib import Path

# Les tests de l'orchestrateur vérifient le graphe avec les stubs (scénarios simulés : drift, pannes...).
# Les vrais agents ont leurs propres tests dans agents/<agent>/tests.
os.environ.setdefault("ORCH_AGENTS", "stub")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not (ROOT / "eval" / "projects" / "variants").exists():
    subprocess.run([sys.executable, str(ROOT / "eval" / "make_variants.py")], check=True)
