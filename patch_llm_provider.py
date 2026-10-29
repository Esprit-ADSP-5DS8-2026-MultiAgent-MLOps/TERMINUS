"""Patch idempotent de agentic_core/llm_client.py (lancer depuis la racine du projet : python patch_llm_provider.py).
Fonctionne depuis l'état d'origine OU depuis la 1re version de ce patch (il la corrige).

Fournisseur « compatible OpenAI » (Gemini, Mistral, OpenRouter, Ollama...) sans changer le code :
  LLM_BASE_URL, LLM_API_KEY                   -> pour tous les agents
  LLM_BASE_URL_<AGENT>, LLM_API_KEY_<AGENT>   -> pour un agent précis (AGENT = VALIDATOR, REPAIR, JUDGE)
  LLM_MAX_TOKENS_PARAM = max_completion_tokens | max_tokens | none
        nom du paramètre « longueur max de sortie » envoyé au fournisseur (none = ne pas l'envoyer).
        Non défini : le comportement d'origine est conservé (max_completion_tokens), comme pour Groq.
  les noms de modèles restent GROQ_MODEL_<AGENT> (inchangé)."""
import pathlib, sys

p = pathlib.Path("agentic_core/llm_client.py")
if not p.exists(): sys.exit("lance ce script depuis la racine du projet")
s = p.read_text(encoding="utf-8")
done = []

def sub(old, new, tag):
    global s
    assert s.count(old) == 1, f"ancre introuvable ou ambiguë pour '{tag}' (llm_client.py a changé ?)"
    s = s.replace(old, new, 1); done.append(tag)

if "def _env(" not in s:
    sub("class GroqClient:", 'def _env(name, agent=None):\n    """Variable d\'environnement propre à l\'agent (NOM_AGENT) puis globale (NOM)."""\n'
        '    return (os.environ.get(f"{name}_{agent.upper()}") if agent else None) or os.environ.get(name) or None\n\n\nclass GroqClient:', "helper _env")
if 'api_key or _env("LLM_API_KEY"' not in s:
    sub('self.api_key = api_key or os.environ.get("GROQ_API_KEY")', 'self.api_key = api_key or _env("LLM_API_KEY", agent) or os.environ.get("GROQ_API_KEY")', "clé LLM_API_KEY")
if '_env("LLM_BASE_URL"' not in s:
    sub('self.base_url = (base_url or self.BASE_URL).rstrip("/")', 'self.base_url = (base_url or _env("LLM_BASE_URL", agent) or self.BASE_URL).rstrip("/")', "URL LLM_BASE_URL")

NEW_BLOCK = ('        mt = _env("LLM_MAX_TOKENS_PARAM", self.agent)    # max_completion_tokens (défaut) | max_tokens | none\n'
             '        if mt:\n'
             '            v = payload.pop("max_completion_tokens")\n'
             '            if mt.lower() != "none":\n'
             '                payload[mt] = v\n')
OLD_BLOCK = ('        if "groq.com" not in self.base_url:        # autres fournisseurs compatibles OpenAI : paramètre standard\n'
             '            payload["max_tokens"] = payload.pop("max_completion_tokens")\n')
if OLD_BLOCK in s:
    sub(OLD_BLOCK, NEW_BLOCK, "ancien test groq.com remplacé par LLM_MAX_TOKENS_PARAM")
elif "LLM_MAX_TOKENS_PARAM" not in s:
    sub('        if self.reasoning_effort and self.model.startswith("openai/gpt-oss"):',
        NEW_BLOCK + '        if self.reasoning_effort and self.model.startswith("openai/gpt-oss"):', "LLM_MAX_TOKENS_PARAM")

p.write_text(s, encoding="utf-8")
print("appliqué :", ", ".join(done) if done else "rien (déjà à jour)")