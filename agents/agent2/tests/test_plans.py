import copy
import json

import pytest

from mlflow_agent.plans import InvalidPlanError, check_plan_against_report, parse_plan
from mlflow_agent.schemas import AnalysisReport
from tests.test_mlflow_agent_input import SAMPLE

GOOD = {
    "models": ["logistic_regression", "random_forest"],
    "primary_metric": "f1",
    "search_method": "optuna",
    "max_trials": 20,
    "rationale": "Imbalanced classification, so compare a linear and a tree model.",
}


def report():
    return AnalysisReport.model_validate(copy.deepcopy(SAMPLE))


def plan_with(**changes):
    return parse_plan(json.dumps({**GOOD, **changes}))


def test_good_plan_passes():
    check_plan_against_report(plan_with(), report())


def test_not_json_rejected():
    with pytest.raises(InvalidPlanError):
        parse_plan("Sure! Here is my plan: use XGBoost")


def test_unknown_model_rejected():
    with pytest.raises(InvalidPlanError):
        plan_with(models=["os_system_model"])


def test_extra_field_rejected():
    with pytest.raises(InvalidPlanError):
        parse_plan(json.dumps({**GOOD, "shell": "rm -rf /"}))


def test_regression_model_rejected_for_classification():
    with pytest.raises(InvalidPlanError):
        check_plan_against_report(plan_with(models=["ridge"]), report())


def test_wrong_metric_rejected():
    with pytest.raises(InvalidPlanError):
        check_plan_against_report(plan_with(primary_metric="accuracy"), report())


def test_too_many_trials_rejected():
    with pytest.raises(InvalidPlanError):
        check_plan_against_report(plan_with(max_trials=50), report())  # report allows 30


def test_huge_plan_rejected():
    with pytest.raises(InvalidPlanError):
        parse_plan("x" * 30_000)
