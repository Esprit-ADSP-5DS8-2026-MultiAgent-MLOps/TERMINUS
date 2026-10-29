"""Patch 3 : taille du préchargement adaptée aux fournisseurs à plafond de tokens/minute (Groq gratuit). À appliquer APRÈS patch_validator2.py."""
from pathlib import Path

A = Path("validator_agent/agent.py")
s = A.read_text(encoding="utf-8")
if "_prefetch_cap" in s:
    raise SystemExit("patch 3 déjà appliqué")
if "root_files = [f for f in tb.known_files" not in s:
    raise SystemExit("appliquez d'abord patch_validator2.py")

def rep(old, new):
    global s
    assert old in s, "bloc introuvable :\n" + old[:90]
    s = s.replace(old, new, 1)

rep("""def prefetch_facts(tb: ValidatorToolBox) -> str:""",
    """def _prefetch_cap(client) -> int:
    \"\"\"Taille max (caractères) des faits préchargés : réduite si UN des LLM de la chaîne a un plafond de tokens/minute (Groq gratuit),
    sinon la requête dépasserait ce plafond et serait refusée avant l'appel (too_large).\"\"\"
    stack, tight = [client], False
    while stack:
        c = stack.pop()
        nested = list(getattr(c, "clients", [])) + ([c.inner] if hasattr(c, "inner") else [])
        if nested:
            stack += nested
        elif getattr(c, "tpm_limit", None):
            tight = True
    return 5200 if tight else 16000


def prefetch_facts(tb: ValidatorToolBox, max_chars: int = 16000) -> str:""")
rep("""    budget = 6000                                     # octets de code préchargés au maximum (~1,7 k tokens)
""", "")
rep("""        size = (tb.root / f).stat().st_size
        if size <= budget:
            budget -= size
            run("read_file", {"path": f})""",
    """        size = int((tb.root / f).stat().st_size * 1.15) + 120          # + en-tête et balises du rendu
        if sum(len(p) for p in parts) + size <= max_chars:              # le code n'est préchargé que s'il reste de la place
            run("read_file", {"path": f})""")
rep("prefetch_facts(tb) +", "prefetch_facts(tb, cfg.prefetch_max_chars or _prefetch_cap(client)) +")
rep("    prefetch: bool = True ", "    prefetch_max_chars: Optional[int] = None    # None = automatique selon les plafonds de tokens des LLM\n    prefetch: bool = True ")
A.write_text(s, encoding="utf-8")
print("patch 3 appliqué")