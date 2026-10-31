import os
import subprocess
import sys
from pathlib import Path

# Les tests de l'orchestrateur vérifient le graphe avec les stubs (scénarios simulés : drift, pannes...)
# et le superviseur à règles : jamais de vrai LLM, même si le .env du développeur est chargé.
# Les vrais agents ont leurs propres tests dans agents/<agent>/tests.
os.environ["ORCH_AGENTS"] = "stub"
os.environ["ORCH_SUPERVISOR"] = "rules"
# Base des runs : toujours SQLite dans un dossier temporaire, même si ORCH_DATABASE_URL (PostgreSQL) est chargée.
os.environ.pop("ORCH_DATABASE_URL", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not (ROOT / "eval" / "projects" / "variants").exists():
    subprocess.run([sys.executable, str(ROOT / "eval" / "make_variants.py")], check=True)
