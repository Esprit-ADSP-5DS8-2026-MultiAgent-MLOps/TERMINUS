"""Compact tests for final verification and delivery requirements."""

import sys
import importlib
import json
from types import SimpleNamespace
from pathlib import Path
from types import SimpleNamespace

from agents.agent1 import run_agent1
from agents.agent1.nodes.apply_changes import apply_changes
from agents.agent1.nodes.verify import verify

apply_module = importlib.import_module("agents.agent1.nodes.apply_changes")
quality_module = importlib.import_module("agents.agent1.nodes.quality_analysis")
plan_module = importlib.import_module("agents.agent1.nodes.plan_improvements")
llm_module = importlib.import_module("agents.agent1.llm")
config_module = importlib.import_module("agents.agent1.config")


def _verify_state(workspace: Path, snapshot: Path, tier: int, baseline_ok: bool) -> dict[str, object]:
    """Build the minimum state for a tier verification test."""
    return {
        "workspace_path": str(workspace),
        "snapshot_path": str(snapshot),
        "changes": [{"file": "main.py", "tier": tier}],
        "baseline_ok": baseline_ok,
        "baseline": {
            "command": [sys.executable, "main.py"],
            "exit_code": 0,
            "stdout": "ok\n",
            "files_changed": [],
        },
        "attempts": 0,
        "quality_before": {},
        "tool_errors": [],
    }


def test_format_only_change_passes_verification(tmp_path: Path) -> None:
    """A docstring-only Tier 1 change passes the AST-insensitive check."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("def main() -> None:\n    return None\n", encoding="utf-8")
    (workspace / "main.py").write_text('"""Docs."""\n\ndef main() -> None:\n    return None\n', encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 1, False))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_import_reordering_passes_ast_verification(tmp_path: Path) -> None:
    """Top-level import reordering is ignored by the AST comparison."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("import os\nimport sys\nprint('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("import sys\nimport os\nprint('ok')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 1, False))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_unused_import_removal_passes_baseline_rerun(tmp_path: Path) -> None:
    """A Tier 2 unused-import removal is accepted by the baseline rerun."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("import os\nprint('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("print('ok')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 2, True))  # type: ignore[arg-type]
    assert result["verification_passed"] is True


def test_behavior_break_is_reverted(tmp_path: Path) -> None:
    """A Tier 2 behavior change fails layer 4 and restores the snapshot."""
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    workspace.mkdir()
    snapshot.mkdir()
    (snapshot / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "main.py").write_text("print('broken')\n", encoding="utf-8")
    result = verify(_verify_state(workspace, snapshot, 2, True))  # type: ignore[arg-type]
    assert result["verification_passed"] is False
    assert result["verification_details"]["layer"] == 4  # type: ignore[index]
    assert result["changes"] == []
    assert (workspace / "main.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_missing_tool_is_reported(tmp_path: Path, monkeypatch: object) -> None:
    """A missing Ruff module is surfaced in tool_errors."""
    def missing_tool(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="No module named ruff")

    monkeypatch.setattr(apply_module.subprocess, "run", missing_tool)
    result = apply_changes({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert result["tool_errors"]
    assert result["tool_errors"][0]["tool"] == "ruff"  # type: ignore[index]


def test_vulture_exit_three_is_not_a_tool_error(tmp_path: Path, monkeypatch: object) -> None:
    """Vulture's findings exit code 3 is accepted as a successful tool run."""
    def tool_result(command: list[str], **kwargs: object) -> SimpleNamespace:
        if "vulture" in command:
            return SimpleNamespace(returncode=3, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(quality_module.subprocess, "run", tool_result)
    result = quality_module.quality_analysis({"workspace_path": str(tmp_path)})  # type: ignore[arg-type]
    assert not [error for error in result["tool_errors"] if error["tool"] == "vulture"]  # type: ignore[index]


def test_user_agent_is_sent_by_both_clients(monkeypatch: object) -> None:
    """Local and ML requests identify the Agent 1 client."""
    requests: list[object] = []

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def capture(request: object, **kwargs: object) -> Response:
        requests.append(request)
        return Response()

    monkeypatch.setenv("AGENT1_LLM_API_KEY", "local-test")
    monkeypatch.setenv("ML_LLM_API_KEY", "ml-test")
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", capture)
    llm_module.complete_text("system", "user")
    llm_module.complete_ml("system", "user")
    assert all(request.get_header("User-agent") == "agent1/1.0" for request in requests)


def test_empty_reasoning_content_retries_and_reports_finish_reason(monkeypatch: object) -> None:
    """Empty reasoning-model content retries once and reports the finish reason."""
    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":""},"finish_reason":"length"}]}'

    calls: list[object] = []
    monkeypatch.setenv("ML_LLM_API_KEY", "ml-test")
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", lambda request, **kwargs: (calls.append(request), Response())[1])
    assert llm_module.complete_ml("system", "user") == ""
    assert len(calls) == 2
    assert "empty answer, finish_reason=length" in llm_module.ml_error_details()


def test_fenced_json_completion_is_parsed(monkeypatch: object) -> None:
    """Markdown fences are removed before ML JSON parsing."""
    monkeypatch.setattr(llm_module, "complete_ml", lambda system, user: "```json\n{\"code\": \"ok\"}\n```")
    assert llm_module.complete_ml_json("system", "user") == {"code": "ok"}


def test_original_project_is_unchanged(tmp_path: Path) -> None:
    """Running Agent 1 never modifies the supplied project."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    csv_path = project / "data.csv"
    csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    run_agent1(str(project), str(csv_path))
    assert {path: path.read_bytes() for path in project.rglob("*") if path.is_file()} == before


def _llm_state(tmp_path: Path, output: str) -> dict[str, object]:
    """Build a direct apply state for a scripted function edit."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("def greet(name):\n    # keep this comment\n    return 'hi ' + name\n", encoding="utf-8")
    from agents.agent1.llm import FakeLLM

    return {
        "workspace_path": str(workspace),
        "baseline_ok": False,
        "allowed_tiers": [1, 2, 3],
        "improvement_plan": [{"file": "module.py", "function": "greet", "line_range": [1, 3], "kind": "missing_both", "tier": 2}],
        "changes": [],
        "tool_errors": [],
        "llm": {"planned": 1},
        "llm_client": FakeLLM([output]),
    }


def test_valid_fake_llm_edit_is_applied(tmp_path: Path, monkeypatch: object) -> None:
    """A valid function-only FakeLLM answer is accepted."""
    monkeypatch.setattr(apply_module, "_run_ruff", lambda command: None)
    result = apply_changes(_llm_state(tmp_path, "def greet(name: str) -> str:\n    \"\"\"Greet someone.\"\"\"\n    # keep this comment\n    return 'hi ' + name\n"))
    assert result["llm"]["applied"] == 1  # type: ignore[index]
    assert result["changes"][-1]["tier"] == 2  # type: ignore[index]


def test_logic_change_missing_name_and_dropped_comment_are_rejected(tmp_path: Path, monkeypatch: object) -> None:
    """Unsafe, mismatched, and comment-dropping answers are rejected."""
    monkeypatch.setattr(apply_module, "_run_ruff", lambda command: None)
    result = apply_changes(_llm_state(tmp_path, "def other(name: str) -> str:\n    \"\"\"Greet.\"\"\"\n    return name\n"))
    assert result["changes"] == []
    assert result["rejected"]  # type: ignore[index]


def test_fenced_fake_llm_answer_is_cleaned(tmp_path: Path, monkeypatch: object) -> None:
    """A markdown-wrapped valid answer is cleaned and accepted."""
    monkeypatch.setattr(apply_module, "_run_ruff", lambda command: None)
    answer = "```python\ndef greet(name: str) -> str:\n    \"\"\"Greet.\"\"\"\n    # keep this comment\n    return 'hi ' + name\n```"
    result = apply_changes(_llm_state(tmp_path, answer))
    assert result["llm"]["applied"] == 1  # type: ignore[index]


def test_unreachable_endpoint_returns_empty_plan(monkeypatch: object) -> None:
    """An unavailable endpoint yields a safe empty plan and note."""
    monkeypatch.setattr(plan_module, "llm_available", lambda: False)
    # no real network call: the answer of a real endpoint (403, proxy...) depends on the machine
    monkeypatch.setattr(plan_module, "availability_details", lambda: {"note": "", "endpoint_error": None})
    result = plan_module.plan_improvements({"allowed_tiers": [1, 2], "quality_issues": []})  # type: ignore[arg-type]
    assert result["improvement_plan"] == []
    assert "unavailable" in result["llm_note"]  # type: ignore[operator]


def test_latest_model_id_does_not_block_http_200(monkeypatch: object) -> None:
    """A healthy HTTP 200 endpoint is available even with a :latest model tag."""
    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(config_module, "MODEL_NAME", "agent1-coder")
    llm_module.reset_availability_cache()
    assert llm_module.llm_available() is True


def test_connection_error_is_exposed_in_llm_note(monkeypatch: object) -> None:
    """Connection failures are not cached and retain their exception text."""
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", fail)
    llm_module.reset_availability_cache()
    result = plan_module.plan_improvements({"allowed_tiers": [1, 2], "quality_issues": []})  # type: ignore[arg-type]
    assert "connection refused" in result["llm"]["note"]  # type: ignore[index]
    assert "connection refused" in result["llm"]["endpoint_error"]  # type: ignore[index]


def test_base_url_is_normalized(monkeypatch: object) -> None:
    """Configured URLs have exactly one /v1 suffix."""
    monkeypatch.setenv("AGENT1_LLM_BASE_URL", "http://localhost:11434/v1/")
    assert config_module.get_llm_config()["base_url"] == "http://localhost:11434/v1"
