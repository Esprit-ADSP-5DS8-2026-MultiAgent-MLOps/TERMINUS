"""Tests for target and feature inference."""

from pathlib import Path

from agents.agent1.nodes.analyze_data import analyze_data
from agents.agent1.nodes.detect_problem_type import detect_problem_type


def _profile(csv_text: str, workspace: Path, target: str | None = None) -> dict[str, object]:
    csv_path = workspace / "data.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    if target:
        (workspace / "mlops.yaml").write_text(
            f"data:\n  target: {target}\n", encoding="utf-8"
        )
    return analyze_data({"csv_path": str(csv_path)})  # type: ignore[arg-type]


def test_manifest_target_and_id_are_honored(tmp_path: Path) -> None:
    """Manifest target wins and row identifiers are excluded from features."""
    profiled = _profile(
        "row_id,age,city,target\n1,20,A,0\n2,30,B,1\n3,40,A,0\n4,50,B,1\n",
        tmp_path,
        target="target",
    )
    result = detect_problem_type(
        {"workspace_path": str(tmp_path), "profile": profiled["profile"]}  # type: ignore[arg-type]
    )
    assert result["target"] == "target"
    assert result["problem_type"] == "classification"
    assert result["subtype"] == "binary"
    assert result["features"] == {"numeric": ["age"], "categorical": ["city"]}
    assert result["confidence"] == 0.95


def test_code_target_and_regression_are_detected(tmp_path: Path) -> None:
    """A code reference can identify a high-cardinality numeric regression target."""
    (tmp_path / "train.py").write_text(
        "TARGET = 'price'\ndf['price']\ny = df[TARGET]\n", encoding="utf-8"
    )
    values = "\n".join(f"{index},{index * 10}" for index in range(1, 21))
    profiled = _profile(f"feature,price\n{values}\n", tmp_path)
    result = detect_problem_type(
        {"workspace_path": str(tmp_path), "profile": profiled["profile"]}  # type: ignore[arg-type]
    )
    assert result["target"] == "price"
    assert result["problem_type"] == "regression"
    assert result["subtype"] is None
    assert result["features"] == {"numeric": ["feature"], "categorical": []}
    assert result["confidence"] == 0.90


def test_missing_profile_is_reported(tmp_path: Path) -> None:
    """Invalid profile data is converted to an error state update."""
    result = detect_problem_type({"workspace_path": str(tmp_path), "profile": {}})  # type: ignore[arg-type]
    assert result["target"] is None
    assert result["confidence"] == 0.0
    assert result["errors"]  # type: ignore[index]
