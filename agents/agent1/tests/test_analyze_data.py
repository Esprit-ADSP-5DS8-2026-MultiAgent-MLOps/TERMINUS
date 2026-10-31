"""Tests for CSV profiling."""

from pathlib import Path

from agents.agent1.nodes.analyze_data import analyze_data


def test_analyze_data_profiles_columns(tmp_path: Path) -> None:
    """The profile includes row counts and per-column statistics."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("age,label,city\n10,0,A\n,1,B\n10,0,A\n", encoding="utf-8")
    result = analyze_data({"csv_path": str(csv_path)})  # type: ignore[arg-type]
    profile = result["profile"]
    assert profile["rows"] == 3
    assert profile["columns"]["age"]["missing"] == 1
    assert profile["columns"]["age"]["unique_count"] == 1
    assert profile["columns"]["age"]["numeric"] is True
    assert profile["columns"]["city"]["dtype"] in {"object", "str"}


def test_analyze_data_records_read_errors(tmp_path: Path) -> None:
    """An unreadable path returns an error update instead of raising."""
    result = analyze_data({"csv_path": str(tmp_path / "missing.csv")})  # type: ignore[arg-type]
    assert result["profile"] == {}
    assert result["errors"]  # type: ignore[index]
