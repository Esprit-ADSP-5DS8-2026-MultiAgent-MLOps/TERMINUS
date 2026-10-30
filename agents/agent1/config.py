"""Configuration for Agent 1."""

BASELINE_TIMEOUT_SECONDS = 300
BASELINE_OUTPUT_MAX_CHARS = 10_000
CLASSIFICATION_MAX_UNIQUE = 10
CLASSIFICATION_UNIQUE_RATIO = 0.2
MAX_CYCLOMATIC_COMPLEXITY = 10
IMBALANCE_RATIO = 0.2
TARGET_COMMON_NAMES = ("target", "label", "y", "class", "price", "default", "churn")
ID_NAME_PARTS = ("id", "index")
ISSUE_TIER_BY_TOOL = {"ruff": 1, "radon": 2, "vulture": 3}
MAX_ATTEMPTS = 3
OUTPUT_TOLERANCE = 1e-6
LLM_MODEL = "groq"
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-20b"
API_KEY = ""
LLM_TIMEOUT = 120
MAX_LLM_ITEMS = 10
ENABLE_LLM = True
TIER3_ENABLED = False
ML_SEED = 42
ENABLE_ML_FIXES = True
ML_LLM_BASE_URL = "https://api.groq.com/openai/v1"
ML_LLM_MODEL = "openai/gpt-oss-120b"
ML_LLM_MAX_TOKENS = 4000
ENV_PATH = __file__
base_url = BASE_URL
model_name = MODEL_NAME
api_key = API_KEY
llm_timeout = LLM_TIMEOUT
max_llm_items = MAX_LLM_ITEMS
enable_llm = ENABLE_LLM
enable_ml_fixes = ENABLE_ML_FIXES
ml_seed = ML_SEED
ml_llm_base_url = ML_LLM_BASE_URL
ml_llm_max_tokens = ML_LLM_MAX_TOKENS
_env_loaded = False


def load_env_file() -> None:
	"""Load package-local KEY=VALUE settings without overriding the environment."""
	global _env_loaded
	if _env_loaded:
		return
	_env_loaded = True
	import os
	from pathlib import Path

	path = Path(ENV_PATH).resolve().with_name(".env")
	try:
		lines = path.read_text(encoding="utf-8").splitlines()
	except FileNotFoundError:
		return
	for line in lines:
		stripped = line.strip()
		if not stripped or stripped.startswith("#") or "=" not in stripped:
			continue
		key, value = stripped.split("=", 1)
		key, value = key.strip(), value.strip()
		if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
			value = value[1:-1]
		value = value.strip(" \t")
		if key and key not in os.environ:
			os.environ[key] = value


def get_llm_config() -> dict[str, object]:
	"""Read lazy environment overrides for the Groq-compatible client."""
	import os
	load_env_file()

	configured_url = os.getenv("AGENT1_LLM_BASE_URL", BASE_URL).strip(" \t").rstrip("/")
	if not configured_url.endswith("/v1"):
		configured_url = f"{configured_url}/v1"
	return {
				"base_url": configured_url.strip(" \t"),
		"model_name": os.getenv("AGENT1_LLM_MODEL", MODEL_NAME),
		"api_key": os.getenv(
			"AGENT1_LLM_API_KEY",
			os.getenv("GROQ_API_KEY", os.getenv("XAI_API_KEY", API_KEY)),
		),
		"llm_timeout": int(os.getenv("AGENT1_LLM_TIMEOUT", str(LLM_TIMEOUT))),
		"max_llm_items": int(os.getenv("AGENT1_MAX_LLM_ITEMS", str(MAX_LLM_ITEMS))),
		"enable_llm": os.getenv("AGENT1_ENABLE_LLM", str(ENABLE_LLM)).lower() in {"1", "true", "yes"},
	}


def get_ml_llm_config() -> dict[str, object]:
	"""Read lazy Groq ML-fix configuration without retaining the API key."""
	import os
	load_env_file()

	return {
				"base_url": os.getenv("ML_LLM_BASE_URL", ML_LLM_BASE_URL).strip(" \t").rstrip("/"),
				"model": os.getenv("ML_LLM_MODEL", ML_LLM_MODEL).strip(" \t"),
				"api_key": os.getenv("ML_LLM_API_KEY", os.getenv("XAI_API_KEY", "")).strip(" \t"),
		"timeout": int(os.getenv("ML_LLM_TIMEOUT", "120")),
		"max_tokens": int(os.getenv("ML_LLM_MAX_TOKENS", str(ML_LLM_MAX_TOKENS))),
		"reasoning_effort": os.getenv("ML_LLM_REASONING_EFFORT", "").strip(" \t"),
	}
