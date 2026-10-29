#!/usr/bin/env python3
"""apply_real_llm_fixes.py : correctifs révélés par le premier essai avec de vrais LLM (Groq).

  1. Judge : une preuve recopiée telle quelle (guillemets, sauts de ligne) était refusée, car les sorties d'outils sont du JSON échappé.
  2. Judge : le modèle « indépendant » était lu AVANT la boucle ; avec une bascule sur un LLM de secours, la mention était fausse.
  3. Judge : modèle par défaut retiré par Groq (llama-3.3-70b-versatile, 16/08/2026) -> qwen/qwen3.6-27b (autre famille que le Repair).
  4. Libellé de refus : affichait toujours « submit_report » ; affiche maintenant l'outil réel (submit_verdict pour le Judge).
  5. Prompt du Repair : un chemin remplacé doit pointer vers un fichier existant (il avait inventé data/credit.csv).

Usage (racine du projet) : python apply_real_llm_fixes.py      Sûr : vérifie toutes les ancres avant d'écrire, refuse de s'exécuter deux fois.
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
EDITS = [
    ("agentic_core/toolkit.py",
     "        self._corpora += [norm(text), norm(stripped)]\n",
     "        self._corpora += [norm(text), norm(stripped)]\n"
     "        # Les sorties d'outils sont du JSON : guillemets et sauts de ligne y sont échappés (\\\" et \\n). Un LLM recopie le texte DÉCODÉ\n"
     "        # (ex. une ligne de diff `+X = \"a\"`) : on accepte aussi les extraits du contenu décodé (toujours un extrait EXACT d'une sortie d'outil).\n"
     "        self._corpora.append(norm(text.replace(\"\\\\n\", \"\\n\").replace('\\\\\"', '\"').replace(\"\\\\\\\\\", \"\\\\\")))\n"
     "        try:\n"
     "            def _walk(o):\n"
     "                if isinstance(o, str):\n"
     "                    self._corpora.append(norm(o))\n"
     "                elif isinstance(o, dict):\n"
     "                    for v in o.values():\n"
     "                        _walk(v)\n"
     "                elif isinstance(o, list):\n"
     "                    for v in o:\n"
     "                        _walk(v)\n"
     "            _walk(json.loads(text))\n"
     "        except ValueError:\n"
     "            pass                                       # sortie tronquée ou non JSON : le corpus brut suffit\n"),
    ("repair_agent/judge/evaluator.py",
     "    v: VerdictSub | None = res.submission\n",
     "    v: VerdictSub | None = res.submission\n"
     "    # avec des LLM de secours, le modèle qui a RÉELLEMENT répondu peut différer du modèle principal : on le relit APRÈS la boucle\n"
     "    judge_model = getattr(client, \"model\", None) or judge_model\n"
     "    independent = (judge_model != base[\"repair_model\"]) if (judge_model and base[\"repair_model\"]) else None\n"
     "    if cfg.require_distinct_model and independent is False:\n"
     "        return closed(FailureKind.INPUT_INVALID, \"après bascule, le Judge a utilisé le même modèle que le Repair (require_distinct_model)\", independent_model=False)\n"),
    ("agentic_core/loop.py",
     "\"submit_report REFUSÉ :\"",
     "f\"{spec.submit_tool} REFUSÉ :\""),
    ("agentic_core/llm_client.py",
     "\"judge\": {\"model\": \"llama-3.3-70b-versatile\"",
     "\"judge\": {\"model\": \"qwen/qwen3.6-27b\""),
    ("repair_agent/prompts/system.md",
     "- Ne triche pas : une réparation doit faire fonctionner le projet pour de vrai, pas masquer le défaut.\n",
     "- Ne triche pas : une réparation doit faire fonctionner le projet pour de vrai, pas masquer le défaut.\n"
     "- Un chemin que tu écris doit pointer vers un fichier qui EXISTE dans le projet (vérifie avec `list_files`, n'invente jamais un dossier). "
     "Si la variable ou le chemin n'est pas utilisé, supprime-le plutôt que d'inventer une valeur.\n"),
]

TEST_FILE = '"""Non-régression : Judge avec de VRAIS LLM (preuves recopiées en texte décodé, modèle réellement utilisé après bascule)."""\nimport sys\nimport tempfile\nfrom pathlib import Path\n\nfrom helpers import ROOT, calls\n\nsys.path.insert(0, str(ROOT / "eval"))\nimport judge_eval  # noqa: E402\n\nfrom agentic_core.llm_client import ScriptedClient  # noqa: E402\nfrom agentic_core.schemas import CRITERIA, RepairReport, seal  # noqa: E402\nfrom repair_agent.judge import JudgeConfig, run_judge  # noqa: E402\nfrom repair_agent.judge.tools import JudgeToolBox  # noqa: E402\nfrom repair_agent.workspace import Workspace  # noqa: E402\n\nD = judge_eval.CASES\n\n\ndef _toolbox():\n    t = Path(tempfile.mkdtemp())\n    s = t / "s"\n    s.mkdir()\n    (s / "train.py").write_text(\'BACKUP = "/home/user/credit.csv"\\nx = 1\\n\')\n    ws = Workspace.create(s, t / "w")\n    (ws.root / "train.py").write_text(\'BACKUP = "data.csv"\\nx = 1\\n\')\n    rr = seal(RepairReport(repair_id="r", created_at="x", project="p", attempt=1, status="REPAIRED", next_action="JUDGE"))\n    tb = JudgeToolBox(s, ws.root, rr)\n    tb.call("get_diff", {"path": "train.py"})\n    return tb\n\n\ndef test_evidence_copied_as_decoded_text_is_grounded():\n    tb = _toolbox()\n    assert tb.is_grounded(\'+BACKUP = "data.csv"\')                                        # guillemets : le JSON les échappe (\\")\n    assert tb.is_grounded(\'-BACKUP = "/home/user/credit.csv"\\n+BACKUP = "data.csv"\')      # plusieurs lignes : le JSON les échappe (\\n)\n\n\ndef test_fabricated_evidence_is_still_refused():\n    tb = _toolbox()\n    assert not tb.is_grounded(\'+BACKUP = "data/credit.csv"\')\n    assert not tb.is_grounded("le diff corrige le chemin")\n\n\nclass SwitchingClient(ScriptedClient):\n    """Simule le LLM de secours : le modèle principal tombe, un autre répond ; `model` reflète le dernier modèle ayant répondu."""\n    def __init__(self, script, first, then):\n        super().__init__(script)\n        self._first, self._then, self._used = first, then, False\n\n    @property\n    def model(self):\n        return self._then if self._used else self._first\n\n    def chat(self, messages, tools=None):\n        self._used = True\n        return super().chat(messages, tools)\n\n\ndef _script():\n    args = {"verdict": "ACCEPT", "next_action": "REVALIDATE", "reason": "r",\n            "criteria": [{"name": n, "result": "PASS", "rationale": "ok", "evidence": "scikit-learn==1.8.0"} for n in CRITERIA]}\n    return calls(("list_changes", {}), ("get_diff", {}), ("get_test_results", {}), ("submit_verdict", args))\n\n\ndef test_independence_uses_the_model_that_actually_answered_after_fallback():\n    d = D / "good_pin_requirements"\n    rr = judge_eval.synthetic_repair_report(d)\n    c = SwitchingClient(_script(), first="judge-primary", then="eval-repair")        # « eval-repair » = modèle du Repair\n    v = run_judge(d / "original", rr, c, workspace=d / "repaired")\n    assert v.judge_model == "eval-repair" and v.independent_model is False\n    assert any("biais" in x for x in v.remaining_concerns)\n\n\ndef test_require_distinct_model_is_enforced_after_fallback_too():\n    d = D / "good_pin_requirements"\n    rr = judge_eval.synthetic_repair_report(d)\n    c = SwitchingClient(_script(), first="judge-primary", then="eval-repair")\n    v = run_judge(d / "original", rr, c, cfg=JudgeConfig(require_distinct_model=True), workspace=d / "repaired")\n    assert v.failure and v.verdict != "ACCEPT"\n'
TEST_PATH = "tests/test_judge_real_llm_fixes.py"


def main() -> int:
    if (ROOT / TEST_PATH).exists():
        print(f"déjà appliqué ({TEST_PATH} existe) : rien à faire")
        return 0
    texts = {}
    for rel, old, new in EDITS:
        p = ROOT / rel
        if not p.is_file():
            print(f"ERREUR : {rel} introuvable. Lancez ce script depuis la racine du projet.")
            return 1
        t = texts.setdefault(rel, p.read_text(encoding="utf-8"))
        if t.count(old) != 1:
            print(f"ERREUR : ancre introuvable ou ambiguë dans {rel} (trouvée {t.count(old)} fois) : {old[:60]!r}\nAucun fichier modifié.")
            return 1
        texts[rel] = t.replace(old, new)
    for rel, t in texts.items():
        (ROOT / rel).write_text(t, encoding="utf-8")
        print(f"modifié : {rel}")
    (ROOT / TEST_PATH).write_text(TEST_FILE, encoding="utf-8")
    print(f"créé    : {TEST_PATH}\nOK. Lancez : pytest -q tests/test_judge_real_llm_fixes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
