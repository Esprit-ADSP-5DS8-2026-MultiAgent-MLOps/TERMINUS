"""Interprétation, en Python pur, de ce que l'interpréteur dit du projet : paquets installés, signatures, évènements d'audit.

AUCUNE liste de bibliothèques ici. Tout vient de trois sources observables :
  - `importlib.metadata.packages_distributions()` : quel paquet installé fournit quel import ;
  - `inspect.signature` : une fonction/classe accepte-t-elle une graine, avec quel défaut ;
  - le hook d'audit (PEP 578) : ce que le code a RÉELLEMENT tenté (réseau, processus), même dissimulé à l'AST.
Le script exécuté dans le venv du projet est `introspect_script.py` ; ce module ne fait qu'interpréter son rapport.
"""
from __future__ import annotations

import re
from functools import lru_cache

SEED_PARAM = re.compile(r"(^|_)(seed|random_state|rng)$")      # nom de PARAMÈTRE (convention d'API), pas un nom de bibliothèque
SEED_CALL = re.compile(r"(^|_)seed$")                           # random.seed, numpy.random.seed, torch.manual_seed, set_seed…
MAX_SYMBOLS = 300


def norm(name) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


@lru_cache(maxsize=1)
def _host_table() -> dict:
    try:
        from importlib.metadata import packages_distributions
        return dict(packages_distributions())
    except Exception:
        return {}


def dist_for_import(mod: str) -> str:
    """Distribution probable d'un import, d'après les paquets installés CHEZ LE VALIDATEUR (métadonnées, sans rien importer) ;
    sinon le nom de l'import. Estimation de première ligne : l'introspection du venv du projet fait foi (`module_dists`)."""
    ds = sorted(_host_table().get(mod) or [])
    return norm(ds[0]) if ds else norm(mod)


def build_request(facts: dict) -> dict | None:
    mods = facts.get("_third_party_modules") or {}
    if not mods:
        return None
    syms = sorted({c["symbol"] for c in facts.get("_third_party_calls") or []})[:MAX_SYMBOLS]
    return {"modules": sorted(mods), "symbols": syms}


def module_dists(report: dict, mod: str) -> list[str]:
    return sorted({norm(d) for d in (report.get("modules") or {}).get(mod, [])})


def import_status(mods, report: dict, declared) -> tuple[list[dict], list[str]]:
    """(installés mais non déclarés directement, introuvables dans l'environnement construit depuis les dépendances déclarées)."""
    declared_n = {norm(d) for d in declared}
    undeclared, unresolved = [], []
    for m in sorted(mods):
        ds = module_dists(report, m)
        if not ds:
            unresolved.append(m)
        elif not set(ds) & declared_n:
            undeclared.append({"module": m, "dists": ds})
    return undeclared, unresolved


def seed_suspects(calls, report: dict, global_seed: bool) -> list[dict]:
    """Appels dont la SIGNATURE accepte une graine (défaut None) et qui n'en reçoivent pas. Présomption, pas une preuve :
    seul le déterminisme MESURÉ (deux entraînements) tranche. `KFold(shuffle=False)` est écarté d'après la signature elle-même."""
    if global_seed:
        return []
    symbols, out, seen = report.get("symbols") or {}, [], set()
    for c in calls:
        s = symbols.get(c["symbol"])
        if not s or s.get("kind") not in ("class", "function") or not s.get("params"):
            continue
        seed = next((p for p in s["params"] if SEED_PARAM.search(p)), None)
        defaults, consts = s.get("defaults") or {}, c.get("consts") or {}
        if not seed or defaults.get(seed) != "None":            # sans graine, ou la bibliothèque en fixe une par défaut
            continue
        if "shuffle" in s["params"] and defaults.get("shuffle") == "False" and consts.get("shuffle") is not True:
            continue                                            # la graine n'agit que si le mélange est activé
        if seed in (c.get("kws") or []) and not (seed in consts and consts[seed] is None):
            continue                                            # fournie (même via une variable)
        if c.get("starkw"):
            continue
        pos = s.get("positional") or []
        if seed in pos and not c.get("star") and c.get("nargs", 0) > pos.index(seed):
            continue                                            # fournie en positionnel
        key = (c["file"], c["line"], c["symbol"])
        if key not in seen:
            seen.add(key)
            out.append({"file": c["file"], "line": c["line"], "call": c["symbol"].rsplit(".", 1)[-1], "symbol": c["symbol"],
                        "param": seed, "default": "None"})
    return out


def unknown_estimator_frameworks(calls, report: dict, in_scope, out_scope) -> dict[str, list[str]]:
    """Paquets dont le code appelle un estimateur (classe avec `fit`) et que le contrat ne classe NI dans le périmètre NI hors
    périmètre. Un utilitaire sans `fit` (tqdm, matplotlib…) n'est pas un framework d'apprentissage : il n'est pas signalé."""
    syms = report.get("symbols") or {}
    in_n, out_n = {norm(x) for x in in_scope}, {norm(x) for x in out_scope}
    found: dict[str, list[str]] = {}
    for sym in sorted({c["symbol"] for c in calls}):
        d = syms.get(sym)
        if not d or not d.get("has_fit"):
            continue
        ds = set(module_dists(report, sym.split(".")[0]))
        if not ds or ds & in_n or ds & out_n:
            continue
        found.setdefault(sorted(ds)[0], []).append(sym)
    return {k: v[:3] for k, v in sorted(found.items())}


def audit_findings(audit: dict | None) -> dict[str, list[dict]]:
    """Évènements d'audit répartis : sanctionnables (appelant = code du candidat) ou simplement signalés (appelant = bibliothèque)."""
    ev = (audit or {}).get("events") or []
    cand = [e for e in ev if e.get("origin") == "candidate"]
    return {"network": [e for e in cand if e.get("kind") == "network"],
            "process": [e for e in cand if e.get("kind") == "process"],
            "library": [e for e in ev if e.get("origin") != "candidate"]}


def describe_event(e: dict) -> str:
    return f"{e.get('event')} {e.get('detail')} ({e.get('where')})"
