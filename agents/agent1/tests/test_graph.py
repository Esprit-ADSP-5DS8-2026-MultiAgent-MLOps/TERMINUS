"""Tests for the Agent 1 graph."""

from pathlib import Path

from agents.agent1 import build_graph, run_agent1


def test_graph_compiles_and_contains_expected_nodes() -> None:
    """The graph compiles and exposes the documented workflow nodes."""
    app = build_graph()
    graph_nodes = set(app.get_graph().nodes)
    assert {
        "load_project",
        "capture_baseline",
        "quality_analysis",
        "plan_improvements",
        "apply_changes",
        "verify",
        "retry",
        "revert_and_continue",
        "analyze_data",
        "detect_problem_type",
        "build_output",
    } <= graph_nodes


def test_graph_runs_end_to_end(tmp_path: Path) -> None:
    """The graph produces the output contract shape end to end."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    csv_path = project / "data.csv"
    csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    result = run_agent1(str(project), str(csv_path))
    assert result["code_quality"]["verified"] is True
    assert result["problem_type"] == "classification"
    assert result["subtype"] == "binary"
    assert result["target"] == "target"
    assert result["features"] == {"numeric": ["feature"], "categorical": []}
