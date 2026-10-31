import sys
from pathlib import Path

# agents/agent3/ (le paquet agent_api_docker) et la racine du dépôt (common/, orchestrator/) importables
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2])]
