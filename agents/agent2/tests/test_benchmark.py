import json

import pytest

from mlflow_agent.benchmark import CHECKS, make_cases, run_benchmark, score_output, summarize
from mlflow_agent.llm_client import FakeClient
from mlflow_agent.prompts import SYSTEM_PROMPT

GOOD = {
    "models": ["logistic_regression", "random_forest"],
    "primary_metric": "f1",
    "search_method": "optuna",
    "max_trials": 10,
    "rationale": "ok",
}


def case():
    return next(c for c in make_cases() if c.id == "cls-f1-20000-imb-b30")


def test_cases_are_unique_and_enough():
    cases = make_cases()
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases)


def test_good_plan_passes_all_checks():
    checks = score_output(json.dumps(GOOD), case().report)
    assert all(checks.values())


@pytest.mark.parametrize("raw,failing", [
    ("not json at all", "json_valid"),
    (json.dumps({"models": ["x"]}), "schema_valid"),
    (json.dumps({**GOOD, "models": ["ridge"]}), "models_fit_task"),
    (json.dumps({**GOOD, "models": ["random_forest", "random_forest"]}), "no_duplicates"),
    (json.dumps({**GOOD, "primary_metric": "accuracy"}), "metric_correct"),
    (json.dumps({**GOOD, "max_trials": 31}), "within_budget"),  # budget is 30
])
def test_each_failure_is_detected(raw, failing):
    checks = score_output(raw, case().report)
    assert checks[failing] is False


def test_runner_and_summary_with_fake_client():
    client = FakeClient([json.dumps(GOOD), "garbage", json.dumps(GOOD)])
    results = run_benchmark(client, SYSTEM_PROMPT, make_cases(), limit=3)
    assert len(results) == 3
    s = summarize(results)
    assert s["cases"] == 3 and set(s["checks"]) == set(CHECKS)
    assert results[1]["all_passed"] is False


def test_llm_error_is_recorded_not_raised():
    results = run_benchmark(FakeClient([]), SYSTEM_PROMPT, make_cases(), limit=1)
    assert results[0]["error"] and results[0]["all_passed"] is False
