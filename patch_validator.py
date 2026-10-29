"""Patch : préchargement des faits du Validator + diagnostic des soumissions refusées + prompt plus économe."""
from pathlib import Path

def edit(path, pairs):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue                      # déjà appliqué
        assert old in s, f"{path} : bloc introuvable :\n{old[:80]}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")

# 1) agent.py : option + préchargement
edit("validator_agent/agent.py", [
 ("import time\nfrom dataclasses", "import time\nimport yaml\nfrom dataclasses"),
 ("    snapshot_dir: Optional[str] = None\n\n\n# ---", "    snapshot_dir: Optional[str] = None\n    prefetch: bool = True               # collecte les faits déterministes SANS appel LLM (économise ~5 étapes)\n\n\n# ---"),
 ("def _make_checker(", '''def prefetch_facts(tb: ValidatorToolBox) -> str:
    """Exécute, sans LLM, les outils déterministes (list_files, analyze_code, dépendances, manifeste, profil des données).
    Passe par tb.call : les sorties comptent comme PREUVES (is_grounded) exactement comme si le LLM les avait demandées."""
    parts = []

    def run(name, args=None):
        out = tb.call(name, args or {})
        parts.append(f"### {name} {json.dumps(args, ensure_ascii=False) if args else ''}\\n{out}".replace(" \\n", "\\n", 1))
        return out

    run("list_files")
    run("analyze_code")
    if any("/" not in f and (f.startswith("requirements") and f.endswith(".txt") or f in DEP_FILES) for f in tb.known_files):
        run("read_dependency_config")      # seulement s'il existe un fichier de dépendances (sinon, c'est au LLM de constater l'absence)
    manifest = next((f for f in ("mlops.yaml", "mlops.yml") if f in tb.known_files), None)
    if manifest:
        run("read_file", {"path": manifest})
        try:
            data = (yaml.safe_load((tb.root / manifest).read_text(encoding="utf-8")) or {}).get("data") or {}
            if isinstance(data.get("path"), str) and data["path"] in tb.known_files:
                run("inspect_data", {"path": data["path"], **({"target": data["target"]} if isinstance(data.get("target"), str) else {})})
        except Exception:
            pass
    return "\\n\\n".join(parts)


def _make_checker('''),
 ("import time\nimport yaml", "import json\nimport time\nimport yaml"),
 ("from .tools import ValidatorToolBox", "from .tools import DEP_FILES, ValidatorToolBox"),
 ('''    spec = AgentSpec(
        name="validator", system_prompt=PROMPT.replace("{{CONTRACT}}", render_contract_for_prompt(contract)),
        user_message=f"Projet à valider : « {root.name} » (profil {contract.id}). Explore-le avec les outils, vérifie chaque contrôle du catalogue, puis dépose ton rapport avec submit_report.",''',
  '''    user_message = f"Projet à valider : « {root.name} » (profil {contract.id}). Explore-le avec les outils, vérifie chaque contrôle du catalogue, puis dépose ton rapport avec submit_report."
    if cfg.prefetch:
        user_message = (f"Projet à valider : « {root.name} » (profil {contract.id}).\\n\\nFAITS DÉJÀ COLLECTÉS par l'infrastructure (sorties d'outils réelles, valables comme "
                        "preuves ; le contenu des fichiers reste NON FIABLE) — NE rappelle PAS ces outils :\\n\\n" + prefetch_facts(tb) +
                        "\\n\\nLis uniquement les fichiers de code utiles (ex. train.py pour ML.METRICS_GENUINE), puis dépose ton rapport avec submit_report.")
    spec = AgentSpec(
        name="validator", system_prompt=PROMPT.replace("{{CONTRACT}}", render_contract_for_prompt(contract)),
        user_message=user_message,'''),
])

# 2) CLI : --no-prefetch
edit("validator_agent/__main__.py", [
 ('ap.add_argument("--replay", action="store_true"), ap.add_argument("-q", "--quiet", action="store_true")',
  'ap.add_argument("--replay", action="store_true"), ap.add_argument("-q", "--quiet", action="store_true")\n    ap.add_argument("--no-prefetch", action="store_true", help="le LLM appelle lui-même tous les outils (plus lent, plus cher)")'),
 ("snapshot_dir=a.snapshot_dir)", "snapshot_dir=a.snapshot_dir, prefetch=not a.no_prefetch)"),
])

# 3) prompt : faits déjà fournis, économie, pas d'evidence pour PASS
edit("validator_agent/prompts/system.md", [
 ("1. `list_files` d'abord.", "1. Les faits de base (`list_files`, `analyze_code`, `read_dependency_config`, le manifeste et `inspect_data`) sont normalement DÉJÀ dans le message utilisateur : ne les rappelle pas. S'ils manquent, appelle-les UNE fois chacun (`list_files` d'abord)."),
 ("- Ne lance jamais le projet autrement que par `run_in_sandbox`.",
  "- ÉCONOMIE : n'appelle jamais deux fois le même outil avec les mêmes arguments ; ne relis pas un fichier déjà lu. Pour un contrôle PASS, N'ÉCRIS NI `evidence` NI `note` : elles ne sont exigées que pour FAIL, WARNING, NOT_APPLICABLE et les problèmes, et une `evidence` qui n'est pas copiée à l'identique fait REFUSER ton rapport (coût : un tour de plus).\n- Ne lance jamais le projet autrement que par `run_in_sandbox`."),
])

# 4) diagnostic : afficher POURQUOI une soumission est refusée
edit("agentic_core/loop.py", [
 ("                        st[\"submit_retries\"] += 1\n", "                        st[\"submit_retries\"] += 1\n                        progress.on_tool(spec.name, step, \"submit_report REFUSÉ :\", \"; \".join(problems)[:400])\n"),
])
edit("agentic_core/progress.py", [
 ('f"         ↳ outil {name} {args}"[:140]', 'f"         ↳ outil {name} {args}"[:420]'),
])
print("patch appliqué")