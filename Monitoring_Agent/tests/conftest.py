import sys
from pathlib import Path

# Monitoring_Agent/ (le package) et la racine du repo (common/) importables sans installation
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent)]
