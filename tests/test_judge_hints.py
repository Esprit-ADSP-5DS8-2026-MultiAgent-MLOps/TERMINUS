"""Judge : extraits de preuve valides proposés au refus, list_files, et erreur de bascule lisible."""
import sys

from helpers import ROOT, calls

sys.path.insert(0, str(ROOT / "eval"))
import judge_eval  # noqa: E402

from agentic_core.llm_client import FallbackClient, LLMError, ScriptedClient  # noqa: E402
from agentic_core.schemas import CRITERIA  # noqa: E402
from repair_agent.judge.evaluator import _checker  # noqa: E402
from repair_agent.judge.tools import JudgeToolBox  # noqa: E402

D = judge_eval.CASES


def _tb():
    d = D / "good_pin_requirements"
    tb = JudgeToolBox(d / "original", d / "repaired", judge_eval.synthetic_repair_report(d))
    for name in ("list_changes", "get_diff", "get_test_results"):
        tb.call(name, {})
    return tb


def _args(evidence):
    return {"verdict": "REVISE", "next_action": "RETRY", "reason": "r",
            "criteria": [{"name": n, "result": "PASS", "rationale": "ok", "evidence": evidence} for n in CRITERIA]}


def test_evidence_hints_are_exact_and_grounded():
    tb = _tb()
    hints = tb.evidence_hints()
    assert hints and all(tb.is_grounded(h) for h in hints)


def test_refusal_lists_valid_excerpts_when_evidence_is_a_paraphrase():
    tb = _tb()
    _, problems = _checker(tb)(_args("le diff corrige bien le chemin"))
    assert any("evidence introuvable" in p for p in problems)
    assert any("Extraits VALIDES" in p for p in problems)
    _, ok = _checker(tb)(_args(tb.evidence_hints()[0]))                       # recopier un extrait proposé suffit
    assert not any("evidence introuvable" in p for p in ok)


def test_judge_can_list_files_of_the_repaired_copy():
    tb = _tb()
    assert "list_files" in [s["function"]["name"] for s in tb.specs()]
    assert "requirements.txt" in tb.call("list_files", {})


def test_fallback_switch_message_shows_the_http_error():
    class Boom:
        model = "primary"

        def chat(self, messages, tools=None):
            raise LLMError("HTTP 404 : model_not_found", "http")

    msgs = []
    FallbackClient([Boom(), ScriptedClient([{"role": "assistant", "content": "ok"}], model="second")], on_switch=msgs.append).chat([], None)
    assert msgs and "404" in msgs[0] and "model_not_found" in msgs[0]
