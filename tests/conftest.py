import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not (ROOT / "eval" / "projects" / "variants").exists():
    subprocess.run([sys.executable, str(ROOT / "eval" / "make_variants.py")], check=True)
