"""Patch 4 : (1) appels d'outils avec des champs null (Groq HTTP 400 tool_use_failed), (2) signature Gemini 3 si absente,
(3) préchargement complet quand le LLM qui répond en premier n'a pas de plafond de tokens. À appliquer après patch_validator3.py."""
from pathlib import Path


def edit(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue                                  # déjà appliqué
        assert old in s, f"{path} : bloc introuvable :\n{old[:90]}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")


if "_prefetch_cap" not in Path("validator_agent/agent.py").read_text(encoding="utf-8"):
    raise SystemExit("appliquez d'abord patch_validator2.py puis patch_validator3.py")

HELPERS = """GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator"   # valeur documentée par Google pour des appels d'outils venus d'un autre modèle


def adapt_messages(messages, gemini: bool):
    # Copie de l'historique adaptée au fournisseur : Gemini 3 EXIGE un thought_signature sur chaque appel d'outil
    # (ceux venus de Groq n'en ont pas) ; les autres fournisseurs rejettent ce champ inconnu.
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if m.get("role") != "assistant" or not calls:
            out.append(m)
            continue
        fixed = []
        for tc in calls:
            tc = dict(tc)
            if gemini:
                extra = dict(tc.get("extra_content") or {})
                google = dict(extra.get("google") or {})
                google.setdefault("thought_signature", GEMINI_DUMMY_SIGNATURE)
                extra["google"] = google
                tc["extra_content"] = extra
            else:
                tc.pop("extra_content", None)
            fixed.append(tc)
        out.append({**m, "tool_calls": fixed})
    return out


def _nullable(schema):
    # Copie du schéma où chaque propriété FACULTATIVE accepte aussi null : gpt-oss envoie parfois null et Groq refuse alors l'appel
    # (HTTP 400 tool_use_failed : « expected string, but got null »).
    if isinstance(schema, list):
        return [_nullable(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: (v if k == "required" else _nullable(v)) for k, v in schema.items()}
    props, req = out.get("properties"), set(out.get("required") or [])
    if isinstance(props, dict):
        for name, p in list(props.items()):
            if name not in req and isinstance(p, dict) and isinstance(p.get("type"), str):
                p = dict(p)
                p["type"] = [p["type"], "null"]
                if "enum" in p:
                    p["enum"] = list(p["enum"]) + [None]
                props[name] = p
    return out


class GroqClient:"""

s = Path("agentic_core/llm_client.py").read_text(encoding="utf-8")
if "def _nullable" not in s:
    if "adapt_messages" in s:                         # signature déjà présente : on n'ajoute que _nullable
        helpers = HELPERS[HELPERS.index("def _nullable"):]
    else:
        helpers = HELPERS
    assert "class GroqClient:" in s
    s = s.replace("class GroqClient:", helpers, 1)
    Path("agentic_core/llm_client.py").write_text(s, encoding="utf-8")

edit("agentic_core/llm_client.py", [
 ("""payload = {"model": self.model, "messages": messages, "temperature": self.temperature,""",
  """payload = {"model": self.model, "messages": adapt_messages(messages, "generativelanguage.googleapis.com" in self.base_url),
                   "temperature": self.temperature,"""),
 ("""            payload.update(tools=tools, tool_choice="auto")""",
  """            payload.update(tools=tools if "generativelanguage.googleapis.com" in self.base_url else _nullable(tools), tool_choice="auto")"""),
])

edit("agentic_core/loop.py", [
 ("""def _make_room(""",
  """def _drop_nulls(x):
    \"\"\"Un null pour un champ facultatif = champ omis (les schémas envoyés à Groq acceptent null, le reste du code non).\"\"\"
    if isinstance(x, dict):
        return {k: _drop_nulls(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_drop_nulls(v) for v in x if v is not None]
    return x


def _make_room("""),
 ("""                        raise json.JSONDecodeError("not an object", "", 0)
""",
  """                        raise json.JSONDecodeError("not an object", "", 0)
                    args = _drop_nulls(args)
"""),
])

edit("validator_agent/agent.py", [
 ("    return 5200 if tight else 16000",
  '    return 5200 if (tight and getattr(client, "tpm_limit", None)) else 16000     # lean seulement si le LLM qui répond EN PREMIER a un plafond'),
])
print("patch 4 appliqué")
