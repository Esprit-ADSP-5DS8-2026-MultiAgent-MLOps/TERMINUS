"""LangGraph wiring for Agent 1."""

from langgraph.graph import END, START, StateGraph

from .config import MAX_ATTEMPTS
from .nodes.analyze_data import analyze_data
from .nodes.apply_changes import apply_changes
from .nodes.build_output import build_output
from .nodes.capture_baseline import capture_baseline
from .nodes.detect_problem_type import detect_problem_type
from .nodes.load_project import load_project
from .nodes.ml_fix import ml_fix
from .nodes.ml_review import ml_review
from .nodes.verify_ml import verify_ml
from .nodes.plan_improvements import plan_improvements
from .nodes.quality_analysis import quality_analysis
from .nodes.revert_and_continue import revert_and_continue
from .nodes.retry import retry
from .nodes.verify import verify
from .state import Agent1State


def route_after_verify(state: Agent1State) -> str:
    """Choose the next node after verification."""
    if state["verification_passed"]:
        return "analyze_data"
    if not state.get("improvement_plan"):
        return "analyze_data"
    if state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "revert_and_continue"


def build_graph():
    """Build and compile the Agent 1 state graph."""
    graph = StateGraph(Agent1State)
    graph.add_node("load_project", load_project)
    graph.add_node("capture_baseline", capture_baseline)
    graph.add_node("quality_analysis", quality_analysis)
    graph.add_node("plan_improvements", plan_improvements)
    graph.add_node("apply_changes", apply_changes)
    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_node("revert_and_continue", revert_and_continue)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("detect_problem_type", detect_problem_type)
    graph.add_node("ml_review", ml_review)
    graph.add_node("ml_fix", ml_fix)
    graph.add_node("verify_ml", verify_ml)
    graph.add_node("build_output", build_output)

    graph.add_edge(START, "load_project")
    graph.add_edge("load_project", "capture_baseline")
    graph.add_edge("capture_baseline", "quality_analysis")
    graph.add_edge("quality_analysis", "plan_improvements")
    graph.add_edge("plan_improvements", "apply_changes")
    graph.add_edge("apply_changes", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "analyze_data": "analyze_data",
            "retry": "retry",
            "revert_and_continue": "revert_and_continue",
        },
    )
    graph.add_edge("retry", "plan_improvements")
    graph.add_edge("revert_and_continue", "analyze_data")
    graph.add_edge("analyze_data", "detect_problem_type")
    graph.add_edge("detect_problem_type", "ml_review")
    graph.add_edge("ml_review", "ml_fix")
    graph.add_edge("ml_fix", "verify_ml")
    graph.add_edge("verify_ml", "build_output")
    graph.add_edge("build_output", END)
    return graph.compile()
