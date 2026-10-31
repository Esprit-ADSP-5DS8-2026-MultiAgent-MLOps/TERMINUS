import re

from mlflow_agent.plans import parse_plan
from mlflow_agent.prompts import PROMPTS, TECHNIQUES


def test_baseline_plus_three_techniques():
    assert list(PROMPTS) == ["v0-basic", "v1-structured", "v2-fewshot", "v3-guard"]
    assert set(TECHNIQUES) == set(PROMPTS)


def test_fewshot_example_answers_are_valid_plans():
    answers = re.findall(r"^Answer: (\{.*\})$", PROMPTS["v2-fewshot"], flags=re.MULTILINE)
    assert len(answers) == 3
    for a in answers:
        parse_plan(a)
