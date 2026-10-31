"""Étape 1 : l'énumération des catégories est dans le schéma envoyé au LLM, et une catégorie inventée reste refusée."""
from typing import get_args

from agentic_core.llm_client import GroqClient
from agentic_core.schemas import Category
from mini_project import REQ_LINE, make_mini_project
from policy import load_contract
from validator_agent.agent import Invariants, _make_checker, prefetch_facts
from validator_agent.tools import ValidatorToolBox


def _toolbox(tmp_path, contract=None):
    return ValidatorToolBox(make_mini_project(tmp_path / "proj"), contract or load_contract("sklearn-tabular"))


def _category_schema(tools):
    sub = next(t for t in tools if t["function"]["name"] == "submit_report")["function"]["parameters"]
    return sub["properties"]["issues"]["items"]["properties"]["category"]


def test_schema_enum_comes_from_category_type(tmp_path):
    cat = _category_schema(_toolbox(tmp_path).specs())
    assert cat["enum"] == list(get_args(Category)) and "other" in cat["enum"] and "security" in cat["enum"]


def test_groq_payload_keeps_valid_values_and_allows_null(tmp_path):
    cat = _category_schema(GroqClient(api_key="k").build_payload([{"role": "user", "content": "x"}], _toolbox(tmp_path).specs())["tools"])
    assert set(get_args(Category)) <= set(cat["enum"]) and None in cat["enum"] and cat["type"] == ["string", "null"]


def test_gemini_payload_enum_unchanged(tmp_path):
    gem = GroqClient(api_key="k", model="g", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                     reasoning_effort=None, tpm_limit=0).build_payload([{"role": "user", "content": "x"}], _toolbox(tmp_path).specs())["tools"]
    assert _category_schema(gem)["enum"] == list(get_args(Category))


def test_invalid_category_refused_with_full_list_and_valid_one_accepted(tmp_path):
    contract = load_contract("sklearn-tabular")
    tb = _toolbox(tmp_path, contract)
    prefetch_facts(tb)
    check = _make_checker(tb, contract, Invariants(require_all_mandatory_reported=False, cross_check_facts=False))
    checks = [{"check_id": contract.checks[0].id, "status": "PASS"}]
    issue = {"check_id": contract.checks[0].id, "description": "d", "evidence": REQ_LINE}
    _, problems = check({"checks": checks, "issues": [{**issue, "category": "ml_pipeline"}]})
    assert problems and all(v in problems[0] for v in get_args(Category))        # le refus liste TOUTES les valeurs autorisées
    _, problems = check({"checks": checks, "issues": [{**issue, "category": "dependencies"}]})
    assert not [p for p in problems if "category" in p]