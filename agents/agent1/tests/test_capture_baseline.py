"""Tests for baseline discovery and execution."""

from pathlib import Path

from agents.agent1.nodes.capture_baseline import capture_baseline, find_entry_point


def test_manifest_run_command_is_used(tmp_path: Path) -> None:
    """A declared run command takes precedence over conventional files."""
    (tmp_path / "mlops.yaml").write_text(
        "run: main.py\nentrypoints:\n  train: train.py\n", encoding="utf-8"
    )
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\nprint('baseline')\nPath('created.txt').write_text('x')\n",
        encoding="utf-8",
    )
    command, entry_point = find_entry_point(str(tmp_path))
    assert command is not None
    assert command[0].endswith("python")
    assert entry_point == "main.py"
    result = capture_baseline({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert result["baseline_ok"] is True
    assert result["allowed_tiers"] == [1, 2, 3]
    assert "created.txt" in result["baseline"]["files_changed"]  # type: ignore[index]
    assert result["baseline"]["stdout"] == "baseline\n"  # type: ignore[index]


def test_conventional_entry_point_is_found(tmp_path: Path) -> None:
    """main.py is selected when the manifest has no runnable command."""
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    command, entry_point = find_entry_point(str(tmp_path))
    assert command == [command[0], "main.py"]
    assert entry_point == "main.py"


def test_failed_baseline_is_non_destructive_to_the_node(tmp_path: Path) -> None:
    """Missing entry points produce a controlled failed baseline."""
    result = capture_baseline({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert result["baseline_ok"] is False
    assert result["allowed_tiers"] == [1]
    assert result["errors"]  # type: ignore[index]
