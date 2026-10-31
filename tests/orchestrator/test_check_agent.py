"""check-agent : les stubs sont conformes au contrat, un agent qui triche est signalé."""

from pathlib import Path

from common.schemas import AgentInput, AgentResult, AgentStatus
from orchestrator.check_agent import check, main

ROOT = Path(__file__).resolve().parents[2]
CREDIT_RISK = ROOT / "eval" / "projects" / "credit_risk"


def test_stubs_respect_the_contract(tmp_path):
    for name in ("agent1", "agent2", "agent3", "agent4"):
        rep = check(name, f"orchestrator.stubs.agents:{name}_stub", CREDIT_RISK, tmp_path / name)
        assert rep.ok, (name, [t for ok, t in rep.checks if not ok])


def test_command_line_reports_a_compliant_agent(tmp_path, capsys):
    assert main("agent3", "orchestrator.stubs.agents:agent3_stub", None, str(tmp_path)) == 0
    assert "Conforme" in capsys.readouterr().out


def test_missing_output_and_file_outside_workdir_are_reported(tmp_path):
    outside = tmp_path / "ailleurs.json"

    def bad_agent2(inp: AgentInput) -> AgentResult:
        outside.write_text("{}")
        return AgentResult(agent="agent2", status=AgentStatus.OK, can_proceed=True,
                           outputs={"rapport": str(outside)})

    rep = check("agent2", bad_agent2, CREDIT_RISK, tmp_path / "out")
    failed = [t for ok, t in rep.checks if not ok]
    assert not rep.ok
    assert any("mlflow_metadata" in t for t in failed)
    assert any("workdir" in t for t in failed)


def test_agent_writing_into_the_project_is_reported(tmp_path):
    def sneaky_agent1(inp: AgentInput) -> AgentResult:
        (Path(inp.project_dir) / "notes.txt").write_text("modifié")
        return AgentResult(agent="agent1", status=AgentStatus.FAILED, can_proceed=False, reasons=["test"])

    rep = check("agent1", sneaky_agent1, CREDIT_RISK, tmp_path / "out")
    assert any("intact" in t for ok, t in rep.checks if not ok)
