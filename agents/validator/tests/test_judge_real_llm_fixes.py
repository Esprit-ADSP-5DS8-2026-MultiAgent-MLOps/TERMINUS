"""Non-régression : Judge avec de VRAIS LLM (preuves recopiées en texte décodé, modèle réellement utilisé après bascule)."""
import sys
import tempfile
from pathlib import Path

from helpers import ROOT, calls

sys.path.insert(0, str(ROOT / "eval"))
import judge_eval  # noqa: E402

from agentic_core.llm_client import ScriptedClient  # noqa: E402
from agentic_core.schemas import CRITERIA, RepairReport, seal  # noqa: E402
from repair_agent.judge import JudgeConfig, run_judge  # noqa: E402
from repair_agent.judge.tools import JudgeToolBox  # noqa: E402
from repair_agent.workspace import Workspace  # noqa: E402

D = judge_eval.CASES


def _toolbox():
    t = Path(tempfile.mkdtemp())
    s = t / "s"
    s.mkdir()
    (s / "train.py").write_text('BACKUP = "/home/user/credit.csv"\nx = 1\n')
    ws = Workspace.create(s, t / "w")
    (ws.root / "train.py").write_text('BACKUP = "data.csv"\nx = 1\n')
    rr = seal(RepairReport(repair_id="r", created_at="x", project="p", attempt=1, status="REPAIRED", next_action="JUDGE"))
    tb = JudgeToolBox(s, ws.root, rr)
    tb.call("get_diff", {"path": "train.py"})
    return tb


def test_evidence_copied_as_decoded_text_is_grounded():
    tb = _toolbox()
    assert tb.is_grounded('+BACKUP = "data.csv"')                                        # guillemets : le JSON les échappe (\")
    assert tb.is_grounded('-BACKUP = "/home/user/credit.csv"\n+BACKUP = "data.csv"')      # plusieurs lignes : le JSON les échappe (\n)


def test_fabricated_evidence_is_still_refused():
    tb = _toolbox()
    assert not tb.is_grounded('+BACKUP = "data/credit.csv"')
    assert not tb.is_grounded("le diff corrige le chemin")


class SwitchingClient(ScriptedClient):
    """Simule le LLM de secours : le modèle principal tombe, un autre répond ; `model` reflète le dernier modèle ayant répondu."""
    def __init__(self, script, first, then):
        super().__init__(script)
        self._first, self._then, self._used = first, then, False

    @property
    def model(self):
        return self._then if self._used else self._first

    def chat(self, messages, tools=None):
        self._used = True
        return super().chat(messages, tools)


def _script():
    args = {"verdict": "ACCEPT", "next_action": "REVALIDATE", "reason": "r",
            "criteria": [{"name": n, "result": "PASS", "rationale": "ok", "evidence": "scikit-learn==1.8.0"} for n in CRITERIA]}
    return calls(("list_changes", {}), ("get_diff", {}), ("get_test_results", {}), ("submit_verdict", args))


def test_independence_uses_the_model_that_actually_answered_after_fallback():
    d = D / "good_pin_requirements"
    rr = judge_eval.synthetic_repair_report(d)
    c = SwitchingClient(_script(), first="judge-primary", then="eval-repair")        # « eval-repair » = modèle du Repair
    v = run_judge(d / "original", rr, c, workspace=d / "repaired")
    assert v.judge_model == "eval-repair" and v.independent_model is False
    assert any("biais" in x for x in v.remaining_concerns)


def test_require_distinct_model_is_enforced_after_fallback_too():
    d = D / "good_pin_requirements"
    rr = judge_eval.synthetic_repair_report(d)
    c = SwitchingClient(_script(), first="judge-primary", then="eval-repair")
    v = run_judge(d / "original", rr, c, cfg=JudgeConfig(require_distinct_model=True), workspace=d / "repaired")
    assert v.failure and v.verdict != "ACCEPT"
