"""Boucle Agent 4 -> orchestrateur : la demande de la gateway est remise une fois, avec ses données ;
les nouvelles données couvrent l'épisode en cours, drift des entrées OU baisse de performance."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from monitoring_agent.pipeline import _gateway_request
from monitoring_agent.runtime.supervisor import Supervisor


def test_gateway_request_is_handed_over_once(tmp_path):
    actions = tmp_path / "target" / "actions"
    actions.mkdir(parents=True)
    data = actions / "new_data_D1.parquet"
    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_parquet(data)
    (actions / "retraining_request.json").write_text(json.dumps({
        "schema_version": "0.2", "model_uri": "models:/p/1", "reason": "performance_drop",
        "new_data_path": str(data), "diagnosis": "[concept_drift] roc_auc 0.64"}))
    target, wd = SimpleNamespace(dir=tmp_path / "target"), tmp_path / "wd"
    wd.mkdir()
    outputs, warnings = {}, []
    assert _gateway_request(target, wd, outputs, warnings)
    req = json.loads(Path(outputs["retraining_request"]).read_text())
    assert Path(outputs["retraining_request"]).parent == wd and req["new_data_path"] == str(wd / data.name)
    assert Path(outputs["new_data"]).is_file() and "concept_drift" in warnings[0]
    assert not (actions / "retraining_request.json").exists()
    assert len(list(actions.glob("retraining_request_*.handed.json"))) == 1
    assert not _gateway_request(target, wd, {}, [])          # déjà remise


def _report(i, drifted=0, degraded=False, labels=True):
    return {"window": {"index": i}, "dataset_drift": {"n_drifted": drifted},
            "performance": {"labels_available": labels, "realized_degraded": degraded}}


def test_new_data_is_the_current_episode_even_without_input_drift(tmp_path):
    # fenêtres 3-4 : ancien drift des entrées ; 5 : calme ; 6-7 : concept drift (perf mesurée en baisse)
    reports = [_report(i) for i in range(3)] + [_report(3, drifted=2), _report(4, drifted=1), _report(5),
                                                 _report(6, degraded=True), _report(7, degraded=True),
                                                 _report(8, labels=False)]
    asked = []
    store = SimpleNamespace(window_reports=lambda n: reports,
                            labelled_frame=lambda a, b, t: asked.append((a, b)) or pd.DataFrame(
                                {"x": [a], "target": [1], "y_pred": [0]}))
    fake = SimpleNamespace(history=50, store=store, target=SimpleNamespace(
        window_size=10, mapping=SimpleNamespace(features=["x"], target="target")))
    out = Supervisor.export_new_data(fake, str(tmp_path / "new.parquet"))
    assert asked == [(61, 70), (71, 80)]                      # seulement l'épisode 6-7
    assert list(pd.read_parquet(out).columns) == ["x", "target"]


def test_concept_scenario_is_a_learnable_rule():
    import random

    from monitoring_agent.runtime.simulator import perturb

    rng, med = random.Random(0), {"amount": 100.0}
    high = [perturb({"amount": 150.0}, "concept", rng, ["amount"], [], 1.0, med)[1] for _ in range(50)]
    low = [perturb({"amount": 50.0}, "concept", rng, ["amount"], [], 1.0, med)[1] for _ in range(50)]
    assert all(high) and not any(low)                         # au-delà de la médiane, la cible s'inverse
    assert perturb({"amount": 150.0}, "concept", rng, ["amount"], [], 0.0, med)[1] is False
