"""Configuration centrale : chemins, budgets, modele LLM.

Le rapport d'analyse est lu depuis (par ordre de priorite) :
  1. l'argument --report de la CLI
  2. la variable d'environnement AGENT_REPORT_PATH
  3. workspaces/<project_id>/analysis_report.json   (mode reel)
  4. eval/expected_reports/<project_id>.analysis_report.json  (mode dev / tests)
  5. eval/projects/<project_id>/expected_reports/<project_id>.analysis_report.json
     ou eval/projects/<project_id>/analysis_report.json         (rapport range dans le projet)
  6. si rien n'est trouve : l'agent LLM (discovery_agent.py) explore le depot pour le retrouver
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# Racine du depot = dossier parent de agent_api_docker/
REPO_ROOT = Path(__file__).resolve().parents[1]

WORKSPACES_DIR = Path(os.environ.get("AGENT_WORKSPACES_DIR", REPO_ROOT / "workspaces"))
EVAL_REPORTS_DIR = REPO_ROOT / "eval" / "expected_reports"
EVAL_PROJECTS_DIR = REPO_ROOT / "eval" / "projects"

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
PROMPTS_DIR = PACKAGE_DIR / "prompts"

REPORT_FILENAME = "analysis_report.json"

# Boucle de correction
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", 6))   # nombre max de corrections LLM par execution
DISCOVERY_MAX_STEPS = int(os.environ.get("AGENT_DISCOVERY_MAX_STEPS", 8))   # budget de l'agent de recherche
BUILD_TIMEOUT_S = int(os.environ.get("AGENT_BUILD_TIMEOUT_S", 600))   # build Docker (entrainement du modele inclus)
HEALTHCHECK_TIMEOUT_S = 60
DOCKER_VERBOSE = os.environ.get("AGENT_DOCKER_VERBOSE", "").lower() in ("1", "true", "yes")   # affiche TOUTES les lignes du build
MAX_DOCKER_ITERATIONS = int(os.environ.get("AGENT_MAX_DOCKER_ITERATIONS", 6))   # corrections LLM max pendant build/run/tests Docker

# --- LLM -------------------------------------------------------------------
# Valeurs lues (par priorite) dans l'environnement, puis dans le fichier .env a la racine.
# Le .env est lu dans un dict : il n'est PAS injecte dans os.environ (donc pas herite
# par les sous-processus qui executent le code des projets analyses).
_DOTENV = dotenv_values(REPO_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


LLM_PROVIDER = _get("LLM_PROVIDER")
LLM_MODEL = _get("LLM_MODEL")
LLM_API_KEY_ENV = "LLM_API_KEY_DOCKER"        # nom de la variable (utile pour les messages d'erreur)
LLM_API_KEY = _get(LLM_API_KEY_ENV)           # la valeur de la cle (peut rester vide pour un modele local)
LLM_BASE_URL = _get("LLM_BASE_URL")           # vide = URL par defaut ; sinon vLLM / Ollama
LLM_TIMEOUT_S = int(_get("LLM_TIMEOUT_S", "60"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0"))   # 0 = reponses reproductibles
LLM_MAX_WAIT_S = float(_get("LLM_MAX_WAIT_S", "15"))    # attente max sur un 429 avant de basculer sur le secours

# LLM de secours (optionnels, jusqu'a 4) : essayes dans l'ordre 1 -> 4 si le precedent est indisponible
# (429 / 5xx / reseau). Variables : LLM_FALLBACK_<n>_PROVIDER, _MODEL, _BASE_URL, _API_KEY_DOCKER.
# (les anciens noms sans numero LLM_FALLBACK_PROVIDER... restent acceptes pour le secours n°1)
MAX_FALLBACKS = 4


def _fallback(n: int) -> dict:
    def g(suffix: str) -> str:
        v = _get(f"LLM_FALLBACK_{n}_{suffix}")
        return v or (_get(f"LLM_FALLBACK_{suffix}") if n == 1 else "")
    key_env = f"LLM_FALLBACK_{n}_API_KEY_DOCKER"
    return {"n": n, "provider": g("PROVIDER"), "model": g("MODEL"), "base_url": g("BASE_URL"),
            "key_env": key_env, "api_key": g("API_KEY_DOCKER")}


LLM_FALLBACKS = [f for f in (_fallback(n) for n in range(1, MAX_FALLBACKS + 1)) if f["model"]]

# --- Docker Hub -------------------------------------------------------------
# Le token n'est jamais passe en argument ni affiche : il est envoye sur l'entree standard de `docker login`.
DOCKERHUB_USERNAME = _get("DOCKERHUB_USERNAME")
DOCKERHUB_TOKEN = _get("DOCKERHUB_TOKEN")          # access token Docker Hub (Account settings > Personal access tokens)
DOCKER_IMAGE_NAMESPACE = _get("DOCKER_IMAGE_NAMESPACE") or DOCKERHUB_USERNAME   # espace de nom du depot (utilisateur ou organisation)

# --- Secrets : jamais transmis aux sous-processus -------------------------------
_SECRET_PREFIXES = ("LLM_", "DOCKER", "ANTHROPIC_", "OPENAI_", "MISTRAL_", "AWS_")
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environnement pour executer du code de projet : sans cles ni tokens."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(_SECRET_PREFIXES) and not k.endswith(_SECRET_SUFFIXES)}
    env.update(extra or {})
    return env


if __name__ == "__main__":  # python -m agent_api_docker.config
    print("fichier .env      :", "trouve" if (REPO_ROOT / ".env").is_file() else "ABSENT", f"({REPO_ROOT / '.env'})")
    print("LLM_PROVIDER      :", LLM_PROVIDER or "(vide)")
    print("LLM_MODEL         :", LLM_MODEL or "(vide)")
    print("LLM_BASE_URL      :", LLM_BASE_URL or "(defaut)")
    print(f"{LLM_API_KEY_ENV:<18}:", f"definie ({len(LLM_API_KEY)} caracteres)" if LLM_API_KEY else "NON DEFINIE")
    for f in LLM_FALLBACKS:
        print(f"secours n°{f['n']:<9}:", f"{f['provider'] or f['base_url']} / {f['model']}",
              "(cle OK)" if f["api_key"] else "(cle ABSENTE)")
    if not LLM_FALLBACKS:
        print("LLM de secours    : (aucun)")