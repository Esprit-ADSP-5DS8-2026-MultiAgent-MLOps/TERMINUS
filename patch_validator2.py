"""Patch : préchargement des faits du Validator + diagnostic des soumissions refusées + prompt plus économe."""
from pathlib import Path

if "root_files = [f for f in tb.known_files" in Path("validator_agent/agent.py").read_text(encoding="utf-8"):
    raise SystemExit("patch déjà appliqué")

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

# ---------------------------------------------------------------- partie 2
# 5) préchargement des fichiers de dépendances BRUTS (la preuve d'un framework = la ligne « pkg==x.y ») et du code racine
edit("validator_agent/agent.py", [
 ("""    return "\\n\\n".join(parts)\n\n\ndef _make_checker(""",
  """    root_files = [f for f in tb.known_files if "/" not in f]
    for f in sorted(root_files):                      # dépendances brutes : sans elles, la preuve « scikit-learn==1.8.0 » est introuvable
        if (f.startswith("requirements") and f.endswith(".txt")) or f in DEP_FILES:
            run("read_file", {"path": f})
    budget = 6000                                     # octets de code préchargés au maximum (~1,7 k tokens)
    for f in sorted((f for f in root_files if f.endswith(".py")), key=lambda n: (n != "train.py", n != "main.py", n)):
        size = (tb.root / f).stat().st_size
        if size <= budget:
            budget -= size
            run("read_file", {"path": f})
    return "\\n\\n".join(parts)\n\n\ndef _make_checker("""),
 ("Lis uniquement les fichiers de code utiles (ex. train.py pour ML.METRICS_GENUINE), puis dépose ton rapport avec submit_report.",
  "Les fichiers Python de la racine et les dépendances sont inclus ci-dessus : ne lis un autre fichier que s'il est indispensable, puis dépose ton rapport avec submit_report."),
 # 6) preuve d'un framework : l'infrastructure retrouve elle-même la vraie ligne au lieu de refuser le rapport
 ("""            if f.evidence and not tb.is_grounded(f.evidence):
                problems.append(f"framework {f.name} : evidence introuvable")""",
  """            if f.evidence and not tb.is_grounded(f.evidence):
                # preuve réelle (ligne de dépendance ou d'import) trouvée par l'infrastructure : remplace la citation approximative du LLM
                alt = next((c for c in (f'"name": "{f.name}"', f"{f.name}==", f"import {f.name}", f"from {f.name} ") if tb.is_grounded(c)), None)
                if alt:
                    f.evidence = alt
                else:
                    problems.append(f"framework {f.name} : evidence introuvable (recopie une ligne EXACTE déjà présente dans les sorties, ou omets `evidence`)")"""),
])

# 7) message de refus plus directif : ne pas relancer d'outils pour corriger une preuve
edit("agentic_core/loop.py", [
 ("""\"instruction\": f\"corrige puis rappelle {spec.submit_tool}\"}""",
  """\"instruction\": f\"corrige puis rappelle {spec.submit_tool} SANS relire de fichier : recopie à l'identique un extrait DÉJÀ présent dans les sorties d'outils de la conversation ; l'evidence d'un framework peut être omise.\"}"""),
])

# 8) prompt : preuve d'un framework
edit("validator_agent/prompts/system.md", [
 ("Une preuve inventée est détectée et refusée.",
  "Une preuve inventée est détectée et refusée. Pour `frameworks[].evidence`, copie la ligne exacte du fichier de dépendances (ex. `scikit-learn==1.8.0`) ou d'import ; en cas de doute, omets `evidence`."),
])

# 9) ordre des fournisseurs : LLM_ORDER=2,groq,1 (2 = FALLBACK_2 d'abord, puis Groq clé principale, puis FALLBACK_1)
edit("agentic_core/llm_client.py", [
 ("""        primary = GroqClient.for_agent(agent, model, max_retries=int(os.environ.get("PRIMARY_MAX_RETRIES", 1)),
                                       max_retry_wait=float(os.environ.get("PRIMARY_MAX_WAIT_S", 5)))
        client = FallbackClient([primary] + [build_fallback_client(agent, s, max_retries=2) for s in specs],""",
  """        named = {"groq": GroqClient.for_agent(agent, model, max_retries=int(os.environ.get("PRIMARY_MAX_RETRIES", 1)),
                                              max_retry_wait=float(os.environ.get("PRIMARY_MAX_WAIT_S", 5)))}
        named.update({str(s["n"]): build_fallback_client(agent, s, max_retries=2) for s in specs})
        wanted = [x.strip().lower() for x in os.environ.get("LLM_ORDER", "").split(",") if x.strip()]
        order = [n for n in dict.fromkeys(wanted) if n in named] + [n for n in named if n not in wanted]   # LLM_ORDER d'abord
        client = FallbackClient([named[n] for n in order],"""),
])
print("patch appliqué")