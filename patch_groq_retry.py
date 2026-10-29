"""Patch idempotent de agentic_core/llm_client.py (lancer depuis la racine du projet : python patch_groq_retry.py).
1) HTTP 400 `tool_use_failed` (Groq n'a pas pu lire l'appel d'outil généré) -> 2 nouvelles tentatives, puis erreur comme avant.
2) Variable d'environnement GROQ_MAX_TOKENS_<AGENT> (ex. GROQ_MAX_TOKENS_VALIDATOR=4000) pour la longueur maximale de sortie.
Rien d'autre ne change : l'échec fermé est conservé si les tentatives échouent."""
import pathlib, sys

p = pathlib.Path("agentic_core/llm_client.py")
if not p.exists(): sys.exit("lance ce script depuis la racine du projet")
s = p.read_text(encoding="utf-8")
done = []

A1 = '            body = (getattr(resp, "text", "") or "")[:300]\n'
R1 = ('            if code == 400 and attempt < min(self.max_retries, 2) and "tool_use_failed" in (getattr(resp, "text", "") or ""):\n'
      '                self._sleep(1.0 + attempt)        # appel d\'outil mal formé côté modèle : souvent non reproductible, on redemande\n'
      '                continue\n')
if "tool_use_failed" not in s:
    assert A1 in s, "ancre 1 introuvable (llm_client.py a changé ?)"
    s = s.replace(A1, R1 + A1, 1); done.append("retry tool_use_failed")

A2 = '    return s\n\n\nclass GroqClient:'
R2 = ('    mt = os.environ.get(f"GROQ_MAX_TOKENS_{agent.upper()}")\n'
      '    if mt and mt.isdigit():\n'
      '        s["max_completion_tokens"] = int(mt)\n')
if "GROQ_MAX_TOKENS_" not in s:
    assert A2 in s, "ancre 2 introuvable (llm_client.py a changé ?)"
    s = s.replace(A2, R2 + A2, 1); done.append("GROQ_MAX_TOKENS_<AGENT>")

p.write_text(s, encoding="utf-8")
print("appliqué :", ", ".join(done) if done else "rien (déjà à jour)")