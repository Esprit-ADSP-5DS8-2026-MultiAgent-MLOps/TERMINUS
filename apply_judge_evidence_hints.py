#!/usr/bin/env python3
"""apply_judge_evidence_hints.py : 2e essai avec de vrais LLM.

  1. Bascule de LLM : le message affiche maintenant l'erreur HTTP (code + corps) -> on voit POURQUOI un modèle est « indisponible ».
  2. Judge : quand une preuve est refusée, le refus liste des extraits VALIDES (vérifiés) à recopier tels quels.
  3. Judge : outil list_files (copie corrigée) pour vérifier qu'un chemin ajouté existe.
  4. Prompt du Judge : l'absence de tests dans le projet n'est pas un motif de REVISE ; format attendu des preuves.

Usage (racine du projet) : python apply_judge_evidence_hints.py      (à lancer APRÈS apply_real_llm_fixes.py)
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
EDITS = [
    ("agentic_core/llm_client.py",
     "self.on_switch(f\"{c.model} indisponible ({e.kind}) -> bascule sur {self.clients[order[rank + 1]].model}\")",
     "self.on_switch(f\"{c.model} indisponible ({e.kind}) : {str(e)[:220]} -> bascule sur {self.clients[order[rank + 1]].model}\")"),
    ("repair_agent/judge/evaluator.py",
     "        if not tb.diff_examined:\n",
     "        if any(c.evidence and not tb.is_grounded(c.evidence) for c in v.criteria) and (h := tb.evidence_hints()):\n"
     "            p.append(\"Extraits VALIDES à recopier tels quels dans `evidence` : \" + \" | \".join(h))\n"
     "        if not tb.diff_examined:\n"),
    ("repair_agent/judge/tools.py",
     "    def specs(self):\n        return [\n            fn_spec(\"list_changes\",",
     "    def evidence_hints(self) -> list[str]:\n"
     "        \"\"\"Extraits EXACTS présents dans les sorties d'outils déjà vues (aide la recopie ; chacun est vérifié par is_grounded).\"\"\"\n"
     "        diff = [l[:80] for l in self.real_diff.splitlines() if l[:1] in (\"+\", \"-\") and not l.startswith((\"+++\", \"---\")) and l[1:].strip()]\n"
     "        tests = [f'\"name\": \"{t.name}\", \"status\": \"{t.status}\"' for t in self.repair.tests]\n"
     "        return [h for h in diff if self.is_grounded(h)][:2] + [h for h in tests if self.is_grounded(h)][:3]\n"
     "\n"
     "    def specs(self):\n        return [\n"
     "            fn_spec(\"list_files\", \"Liste les fichiers de la copie corrigée (chemin, taille) : pour vérifier qu'un chemin ajouté existe.\", {}),\n"
     "            fn_spec(\"list_changes\","),
    ("repair_agent/judge/system.md",
     "Sois concis : quelques appels d'outils, verdict complet.\n",
     "## Ce qui justifie (ou non) un REVISE\n"
     "- L'absence de tests unitaires dans le PROJET n'est pas un défaut de la réparation (le Repair ne peut pas en créer : tests protégés) : "
     "CONCERN sur evidence_quality au plus, jamais un motif de REVISE à lui seul. REVISE n'a de sens que si le Repair PEUT corriger ce que tu reproches.\n"
     "- Un chemin ajouté ou modifié doit exister : vérifie-le avec `list_files` avant de conclure.\n"
     "- `evidence` : recopie EXACTEMENT un extrait, p. ex. une ligne de diff (`+X = \"a\"`) ou `\"name\": \"import\", \"status\": \"PASS\"` ; une paraphrase est refusée.\n"
     "Sois concis : quelques appels d'outils, verdict complet.\n"),
]

TEST_FILE = '"""Judge : extraits de preuve valides proposés au refus, list_files, et erreur de bascule lisible."""\nimport sys\n\nfrom helpers import ROOT, calls\n\nsys.path.insert(0, str(ROOT / "eval"))\nimport judge_eval  # noqa: E402\n\nfrom agentic_core.llm_client import FallbackClient, LLMError, ScriptedClient  # noqa: E402\nfrom agentic_core.schemas import CRITERIA  # noqa: E402\nfrom repair_agent.judge.evaluator import _checker  # noqa: E402\nfrom repair_agent.judge.tools import JudgeToolBox  # noqa: E402\n\nD = judge_eval.CASES\n\n\ndef _tb():\n    d = D / "good_pin_requirements"\n    tb = JudgeToolBox(d / "original", d / "repaired", judge_eval.synthetic_repair_report(d))\n    for name in ("list_changes", "get_diff", "get_test_results"):\n        tb.call(name, {})\n    return tb\n\n\ndef _args(evidence):\n    return {"verdict": "REVISE", "next_action": "RETRY", "reason": "r",\n            "criteria": [{"name": n, "result": "PASS", "rationale": "ok", "evidence": evidence} for n in CRITERIA]}\n\n\ndef test_evidence_hints_are_exact_and_grounded():\n    tb = _tb()\n    hints = tb.evidence_hints()\n    assert hints and all(tb.is_grounded(h) for h in hints)\n\n\ndef test_refusal_lists_valid_excerpts_when_evidence_is_a_paraphrase():\n    tb = _tb()\n    _, problems = _checker(tb)(_args("le diff corrige bien le chemin"))\n    assert any("evidence introuvable" in p for p in problems)\n    assert any("Extraits VALIDES" in p for p in problems)\n    _, ok = _checker(tb)(_args(tb.evidence_hints()[0]))                       # recopier un extrait proposé suffit\n    assert not any("evidence introuvable" in p for p in ok)\n\n\ndef test_judge_can_list_files_of_the_repaired_copy():\n    tb = _tb()\n    assert "list_files" in [s["function"]["name"] for s in tb.specs()]\n    assert "requirements.txt" in tb.call("list_files", {})\n\n\ndef test_fallback_switch_message_shows_the_http_error():\n    class Boom:\n        model = "primary"\n\n        def chat(self, messages, tools=None):\n            raise LLMError("HTTP 404 : model_not_found", "http")\n\n    msgs = []\n    FallbackClient([Boom(), ScriptedClient([{"role": "assistant", "content": "ok"}], model="second")], on_switch=msgs.append).chat([], None)\n    assert msgs and "404" in msgs[0] and "model_not_found" in msgs[0]\n'
TEST_PATH = "tests/test_judge_hints.py"


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
    print(f"créé    : {TEST_PATH}\nOK. Lancez : pytest -q tests/test_judge_hints.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
