"""Tests for the public output contract."""

from agents.agent1.nodes.build_output import build_output


def test_build_output_matches_contract() -> None:
    """Output contains only the documented public result fields."""
    result = build_output(
        {
            "quality_before": {"total": 3},
            "quality_after": {"total": 1},
            "changes": [{"file": "main.py", "description": "formatted", "tier": 1}],
            "verification_passed": True,
            "baseline_ok": True,
            "problem_type": "classification",
            "subtype": "binary",
            "target": "target",
            "features": {"numeric": ["age"], "categorical": []},
            "confidence": 0.95,
        }  # type: ignore[arg-type]
    )
    output = result["output"]
    assert {
        key: output["code_quality"][key]
        for key in ("issues_before", "issues_after", "changes", "verified", "baseline_ok")
    } == {
        "issues_before": 3,
        "issues_after": 1,
        "changes": [{"file": "main.py", "description": "formatted"}],
        "verified": True,
        "baseline_ok": True,
    }
    assert output["problem_type"] == "classification"
    assert output["target"] == "target"
