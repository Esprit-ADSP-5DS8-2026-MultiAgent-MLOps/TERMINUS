# Run from the repository root: python -m streamlit run agents/agent1/app.py
"""Streamlit explainer and runner for Agent 1."""

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agents.agent1 import build_graph, llm
from agents.agent1.runner import _initial_state

ROOT = Path(__file__).resolve().parents[2]
LAST_RUN = Path(__file__).resolve().parent / ".workspace" / "last_run.json"
MAX_TRACE_CHARS = 6000

DESCRIPTIONS = {
    "load_project": "Copies the input project into an isolated workspace and keeps a rollback snapshot.",
    "capture_baseline": "Runs the project's entry point and records its baseline behavior and outputs.",
    "quality_analysis": "Runs deterministic Ruff, Radon, Vulture, and custom code-quality checks.",
    "plan_improvements": "Builds a small deterministic plan for eligible docstring and type-hint edits.",
    "apply_changes": "Applies safe Ruff changes and validates one-function LLM edits when enabled.",
    "verify": "Compiles, imports, compares ASTs, and reruns the baseline where applicable.",
    "retry": "Prepares a more conservative retry after a failed verification.",
    "revert_and_continue": "Restores affected files from the snapshot before continuing to data analysis.",
    "analyze_data": "Profiles CSV columns, dtypes, missing values, and unique values.",
    "detect_problem_type": "Detects the target, ML problem type, subtype, and feature groups.",
    "ml_review": "Reports ML risks without editing the project.",
    "ml_fix": "Applies deterministic fixes, then validated Groq API fixes. These may change results.",
    "verify_ml": "Verifies ML fixes by compiling, importing, and rerunning the baseline.",
    "build_output": "Builds the public JSON result and writes the stable workspace and patch artifacts.",
}


def _install_groq_spy() -> None:
    """Record every ML LLM (Groq) call so the interface can display it.

    The recorder wraps llm.complete_ml from the outside, so no agent code changes.
    The trace lives on the llm module because Streamlit re-runs this file on every
    click and would reset a normal variable. The API key is never recorded.
    """
    if getattr(llm, "_spy_installed", False):
        return
    llm._groq_trace = []
    original = llm.complete_ml

    def spy(system: str, user: str, *args: Any, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {
            "system": str(system)[:MAX_TRACE_CHARS],
            "sent": str(user)[:MAX_TRACE_CHARS],
            "received": None,
            "error": None,
        }
        start = time.time()
        try:
            out = original(system, user, *args, **kwargs)
            text = out if isinstance(out, str) else json.dumps(out, default=str)
            entry["received"] = text[:MAX_TRACE_CHARS] if text else None
            if not text:
                try:
                    entry["error"] = llm.ml_error_details() or "empty answer"
                except Exception:
                    entry["error"] = "empty answer"
            return out
        except Exception as exc:
            entry["error"] = str(exc)[:500]
            raise
        finally:
            entry["seconds"] = round(time.time() - start, 1)
            llm._groq_trace.append(entry)

    llm.complete_ml = spy
    llm._spy_installed = True


_install_groq_spy()


def _groq_config() -> dict[str, Any]:
    """Return the ML LLM settings without any key field."""
    try:
        cfg = llm.get_ml_llm_config()
        return {k: v for k, v in cfg.items() if "key" not in k.lower()}
    except Exception:
        return {}


def _summary(update: dict[str, Any]) -> str:
    """Create a compact plain-English summary for one node update."""
    if "baseline" in update:
        return f"baseline exit code {update['baseline'].get('exit_code', 'unknown')}"
    if "quality_issues" in update:
        return f"{len(update['quality_issues'])} quality issues"
    if "changes" in update:
        return f"{len(update['changes'])} changes"
    if "improvement_plan" in update:
        return f"{len(update['improvement_plan'])} planned edits"
    if "problem_type" in update:
        return f"{update.get('problem_type')} target {update.get('target')}"
    if "output" in update:
        return "final output built"
    if "profile" in update:
        return f"profiled {len(update['profile'].get('columns', {}))} columns"
    if "ml_review" in update:
        return f"{len((update['ml_review'] or {}).get('findings', []))} ML findings"
    if "ml_fixes" in update:
        fixes = update["ml_fixes"] or {}
        return f"{len(fixes.get('applied', []))} ML fixes applied"
    return f"returned {len(update)} fields"


def _render_pipeline() -> None:
    """Render the graph diagram and node explanations."""
    st.subheader("Graph")
    try:
        st.code(build_graph().get_graph().draw_mermaid(), language="mermaid")
    except Exception as exc:
        st.error(f"Could not render graph: {exc}")
    for name, description in DESCRIPTIONS.items():
        with st.expander(f"What does this node do? {name}"):
            st.write(description)


def _render_quality(result: dict[str, Any]) -> None:
    """Render before/after quality metrics and chart (lower is better)."""
    quality = result.get("code_quality", {})
    before = quality.get("quality_before", {})
    after = quality.get("quality_after", {})
    tools = [t for t in ("ruff", "radon", "vulture", "custom") if t in before or t in after]
    shown = tools + (["total"] if "total" in before or "total" in after else [])
    if shown:
        st.caption("Number of issues found. Lower is better, so a green arrow pointing down is an improvement.")
        for col, tool in zip(st.columns(len(shown)), shown):
            now, was = after.get(tool, 0), before.get(tool, 0)
            col.metric(tool, now, now - was, delta_color="inverse")
    if tools:
        frame = pd.DataFrame(
            {"Before": [before.get(t, 0) for t in tools], "After": [after.get(t, 0) for t in tools]},
            index=tools,
        )
        try:
            st.bar_chart(frame, stack=False)
        except TypeError:
            st.bar_chart(frame)


def _render_changes(quality: dict[str, Any], patch: str) -> None:
    """Render the change list, the output locations and the code patch."""
    st.dataframe(pd.DataFrame(quality.get("changes", [])), use_container_width=True)
    paths = {
        "improved_project_path": quality.get("improved_project_path"),
        "patch_path": quality.get("patch_path"),
        "ml_patch_path": (quality.get("ml_fixes") or {}).get("ml_patch_path"),
    }
    st.caption("Where the results are saved (the original project is never modified)")
    st.json(paths)
    st.code(patch or "No patch text saved.", language="diff")


def _render_data(result: dict[str, Any]) -> None:
    """Render the data analysis, the ML review and the ML fixes."""
    quality = result.get("code_quality", {})
    st.write({key: result.get(key) for key in ("problem_type", "subtype", "target", "confidence")})

    st.subheader("ML review (report only)")
    findings = (quality.get("ml_review") or {}).get("findings", [])
    severity_order = {"high": 0, "medium": 1, "low": 2}
    if findings:
        ordered = sorted(findings, key=lambda item: severity_order.get(item.get("severity", "low"), 3))
        st.dataframe(pd.DataFrame(ordered), use_container_width=True)
    else:
        st.info("No ML findings.")

    st.subheader("ML fixes")
    ml_fixes = quality.get("ml_fixes") or {}
    left, right = st.columns(2)
    with left:
        st.write("Applied")
        st.json(ml_fixes.get("applied", []))
    with right:
        st.write("Rejected (tried and failed)")
        st.json(ml_fixes.get("rejected", []))
    if ml_fixes.get("report_only"):
        st.write("Reported only (no automatic fix)")
        st.json(ml_fixes.get("report_only", []))
    if ml_fixes.get("note"):
        st.warning(ml_fixes["note"])
    before_col, after_col = st.columns(2)
    with before_col:
        st.write("Project output before ML fixes")
        st.code(ml_fixes.get("stdout_before") or "(empty)", language="text")
    with after_col:
        st.write("Project output after ML fixes")
        st.code(ml_fixes.get("stdout_after") or "(empty)", language="text")
    ml_patch = ml_fixes.get("ml_patch_path", "")
    if ml_patch and Path(ml_patch).is_file():
        st.write("ML patch")
        st.code(Path(ml_patch).read_text(encoding="utf-8"), language="diff")

    st.subheader("Features")
    st.write("Numeric")
    st.write(result.get("features", {}).get("numeric", []))
    st.write("Categorical")
    st.write(result.get("features", {}).get("categorical", []))


def _render_groq(trace: list[dict[str, Any]]) -> None:
    """Render every Groq API call: what was sent and what came back."""
    st.write(
        "Only the function that contains the problem is sent, with a short instruction. "
        "No CSV data and no API key is sent or shown here. "
        "The agent accepts the answer only if the finding disappears, nothing new appears, "
        "and the project still runs."
    )
    st.write(_groq_config())
    if not trace:
        st.info("No Groq call in this run: no API key, ML fixes switched off, or no finding needed the LLM.")
        return
    st.metric("Groq calls in this run", len(trace))
    for i, call in enumerate(trace, 1):
        label = f"Call {i} | {call.get('seconds', '?')}s" + (" | failed" if call.get("error") else " | answered")
        with st.expander(label, expanded=(i == 1)):
            st.caption("Instructions given to the model")
            st.code(call.get("system", ""), language="text")
            st.caption("Sent: the problem and only the affected function")
            st.code(call.get("sent", ""), language="text")
            st.caption("Received from Groq (before the agent's safety checks)")
            st.code(call.get("received") or "(no answer)", language="json")
            if call.get("error"):
                st.error(call["error"])


def _render_result(result: dict[str, Any], patch: str, trace: list[dict[str, Any]]) -> None:
    """Render all result tabs."""
    quality = result.get("code_quality", {})
    pipeline, code, changes, llm_tab, data, groq_tab, raw = st.tabs(
        ["Pipeline", "Code quality", "Changes", "Local LLM", "Data analysis", "Groq API", "Raw JSON"]
    )
    with pipeline:
        _render_pipeline()
    with code:
        _render_quality(result)
        st.json({key: quality.get(key) for key in ("verified", "baseline_ok", "verification_details", "tool_errors", "failure_reason")})
    with changes:
        _render_changes(quality, patch)
    with llm_tab:
        llm_info = quality.get("llm", {})
        st.write("Groq OSS 20B model (docstrings and type hints)")
        st.write({"model": llm_info.get("model"), "endpoint_ok": llm_info.get("endpoint_ok"), "used": llm_info.get("used"),
                  "planned": llm_info.get("planned"), "applied": llm_info.get("applied")})
        st.subheader("Applied")
        st.json([item for item in quality.get("changes", []) if item.get("tier") == 2])
        st.subheader("Rejected by the safety checks")
        st.json(llm_info.get("rejected", []))
    with data:
        _render_data(result)
    with groq_tab:
        _render_groq(trace)
    with raw:
        st.json(result)


def _run(project_path: str, csv_path: str, use_llm: bool, apply_ml: bool) -> None:
    """Stream one graph run and save its final result, patch and Groq trace."""
    state = _initial_state(project_path, csv_path)
    state["enable_ml_fixes"] = apply_ml
    old_value = os.environ.get("AGENT1_ENABLE_LLM")
    os.environ["AGENT1_ENABLE_LLM"] = "true" if use_llm else "false"
    llm._groq_trace.clear()
    steps = st.empty()
    current: dict[str, Any] = state
    started = time.perf_counter()
    try:
        with st.status("Running Agent 1", expanded=True) as status:
            for update in build_graph().stream(state, stream_mode="updates"):
                for name, node_update in update.items():
                    node_update = node_update if isinstance(node_update, dict) else {}
                    current.update(node_update)
                    elapsed = time.perf_counter() - started
                    steps.info(f"{name} | {elapsed:.2f}s | {_summary(node_update)}")
                    with st.expander(f"What does this node do? {name}"):
                        st.write(DESCRIPTIONS.get(name, "No description available."))
            status.update(label="Agent 1 complete", state="complete")
        result = current.get("output", {})
        patch_path = result.get("code_quality", {}).get("patch_path", "")
        patch = Path(patch_path).read_text(encoding="utf-8") if patch_path and Path(patch_path).is_file() else ""
        trace = list(llm._groq_trace)
        LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN.write_text(
            json.dumps({"result": result, "patch": patch, "groq_trace": trace}, indent=2, default=str),
            encoding="utf-8",
        )
        st.session_state["run"] = (result, patch, trace)
    except Exception as exc:
        st.error(f"Agent 1 failed: {exc}")
    finally:
        if old_value is None:
            os.environ.pop("AGENT1_ENABLE_LLM", None)
        else:
            os.environ["AGENT1_ENABLE_LLM"] = old_value


def main() -> None:
    """Render the Agent 1 dashboard."""
    st.set_page_config(page_title="Agent 1", layout="wide")
    st.title("Agent 1: quality and data analysis")
    with st.sidebar:
        project = st.text_input("Project path", str(ROOT / "eval/projects/credit_risk"))
        csv_path = st.text_input("CSV path", str(ROOT / "eval/projects/credit_risk/data.csv"))
        use_llm = st.checkbox("Use Groq LLM (docstrings, type hints)", value=True)
        apply_ml = st.checkbox("Apply ML fixes (rules + Groq API, changes results)", value=True)
        run = st.button("Run agent", type="primary")
        load = st.button("Load last saved run")
    if run:
        _run(project, csv_path, use_llm, apply_ml)
    if load:
        try:
            saved = json.loads(LAST_RUN.read_text(encoding="utf-8"))
            st.session_state["run"] = (saved["result"], saved.get("patch", ""), saved.get("groq_trace", []))
        except Exception as exc:
            st.error(f"Could not load saved run: {exc}")
    if "run" in st.session_state:
        _render_result(*st.session_state["run"])
    else:
        _render_pipeline()


if __name__ == "__main__":
    main()