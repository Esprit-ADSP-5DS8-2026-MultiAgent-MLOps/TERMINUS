"""Dataset de fine-tuning : scénarios déterministes et cohérents, distillation vérifiée, assemblage MLX."""

from __future__ import annotations

import json

import pytest
import yaml

from monitoring_agent.agent.llm import LLMResponse
from monitoring_agent.datasets import find_file
from monitoring_agent.finetune import build as ft_build
from monitoring_agent.finetune import teacher as ft_teacher


def _available(name: str) -> bool:
    try:
        find_file(name)
        return True
    except FileNotFoundError:
        return False


needs_california = pytest.mark.skipif(not _available("california_housing_reference.csv"), reason="données absentes")


@needs_california
@pytest.mark.parametrize("injection", ["concept", "missing", "unit_bug", "covariate_hard"])
def test_scenarios_are_deterministic_and_labelled_by_the_procedure(injection):
    from monitoring_agent.finetune.scenarios import ACCEPT, make_scenario
    a = make_scenario("california", 3, injection)
    b = make_scenario("california", 3, injection)
    assert a["facts"] == b["facts"] and a["oracle"] == b["oracle"]          # même graine -> même scénario
    assert a["messages"][0]["role"] == "system" and "<faits>" in a["messages"][1]["content"]
    assert "<note_enquete>" not in a["messages"][1]["content"]              # format SLM : faits seuls
    if a["accepted"]:
        assert a["oracle"]["root_cause"] in ACCEPT[injection]
        assert a["rules"]["action"] == a["oracle"]["action"]                # règles = procédure = oracle


def _scenario(action="notify_data_team", cause="data_quality", sid="b-00001", split="train"):
    facts = {"quality.x.missing_values": 0.4, "drift.x.psi": 0.3, "perf.labels_available": 1.0}
    return {"id": sid, "base": "b", "split": split, "injection": "missing", "oracle": {"root_cause": cause, "action": action},
            "signals": ["data_quality"], "features": ["x"], "facts": facts,
            "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "<faits>…</faits>"}],
            "rules": {"root_cause": cause, "summary": "règles", "affected_features": ["x"],
                      "evidence": [{"fact": "quality.x.missing_values", "value": 0.4, "claim": "c"}],
                      "action": action, "rationale": "r", "confidence": 0.8}}


class Teacher:
    """1er appel : mauvaise réponse ; 2e (avec indice) : bonne réponse, valeur citée à 1 % près."""

    def __init__(self):
        self.name, self.seen = "fake:teacher", []

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        self.seen.append(messages[-1]["content"])
        good = "<indice>" in messages[-1]["content"]
        d = {"root_cause": "data_quality" if good else "covariate_shift", "summary": "s", "affected_features": ["x"],
             "evidence": [{"fact": "quality.x.missing_values", "value": 0.401, "claim": "40 % manquants"}],
             "action": "notify_data_team" if good else "request_retraining", "rationale": "r", "confidence": 0.83}
        return LLMResponse(text=json.dumps(d), model=self.name)


def test_teacher_rationalizes_and_training_prompt_has_no_hint(tmp_path, monkeypatch):
    scen = tmp_path / "s.jsonl"
    scen.write_text(json.dumps(_scenario()) + "\n", encoding="utf-8")
    t = Teacher()
    monkeypatch.setattr(ft_teacher, "teachers", lambda spec=None: [t])
    stats = ft_teacher.teach(scen, tmp_path / "t.jsonl", progress=lambda *_: None)
    rec = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))
    assert stats["teacher_rationalized"] == 1 and rec["source"] == "teacher_rationalized"
    assert rec["diagnosis"]["evidence"][0]["value"] == 0.4                   # valeur recopiée exactement
    assert "<indice>" not in t.seen[0] and "<indice>" in t.seen[1]
    # Reprise : relancer ne refait pas le travail
    ft_teacher.teach(scen, tmp_path / "t.jsonl", progress=lambda *_: None)
    assert len(t.seen) == 2

    out = ft_build.build(scen, tmp_path / "t.jsonl", tmp_path / "data")
    ex = json.loads((tmp_path / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert "<indice>" not in ex["messages"][1]["content"]                    # le SLM apprend SANS l'indice
    assert out["train"]["sources"] == {"teacher_rationalized": 1}


def test_build_balances_actions_and_writes_lora_config(tmp_path):
    rows = [_scenario(sid=f"b-{i:05d}") for i in range(200)]
    rows += [_scenario("request_retraining", "concept_drift", sid=f"c-{i:05d}") for i in range(20)]
    rows += [_scenario(sid=f"t-{i:05d}", split="test") for i in range(5)]
    scen = tmp_path / "s.jsonl"
    scen.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = ft_build.build(scen, tmp_path / "absent.jsonl", tmp_path / "data")
    acts = stats["train"]["actions"]
    assert acts["request_retraining"] == 20 and acts["notify_data_team"] == 40       # plafond = 2 × la plus rare
    assert stats["test"]["n"] == 5 and (tmp_path / "data" / "test_cases.jsonl").exists()
    cfg = yaml.safe_load((tmp_path / "data" / "lora.yaml").read_text(encoding="utf-8"))
    assert cfg["mask_prompt"] is True and cfg["lora_parameters"]["rank"] == 16 and cfg["num_layers"] == -1
    assert cfg["iters"] == max(50, 2 * 60 // 4)
