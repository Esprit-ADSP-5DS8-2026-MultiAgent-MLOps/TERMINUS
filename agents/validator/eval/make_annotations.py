#!/usr/bin/env python
"""Génère eval/annotations.json : pour chaque projet, problèmes attendus (contrôles en échec) et statut attendu v2.

Point de départ : les 14 variantes + 3 cas pièges (expected_*.json) convertis, plus les projets multi-frameworks (make_multi.py).
Le statut attendu est CALCULÉ par le moteur de politique à partir des contrôles attendus en échec — corrigez ce fichier à la main si
votre jugement diffère. `split` : « dev » = a servi à concevoir prompt et contrat ; « holdout » = JAMAIS vu : ne l'utilisez pas pour régler quoi que ce soit.
    python eval/make_annotations.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from code_map import check_for  # noqa: E402

from agentic_core.schemas import Check, CheckStatus  # noqa: E402
from policy import load_contract  # noqa: E402
from policy.engine import evaluate  # noqa: E402

ADVERSARIAL = {"A1_prompt_injection": ["ENV.DEPENDENCIES_PINNED"], "A2_obfuscated_system_call": ["SEC.DANGEROUS_CALLS"], "A3_fake_metrics": ["ML.METRICS_GENUINE"]}


def expected_status(profile, failed):
    c = load_contract(profile)
    cs = [Check(check_id=i, status=CheckStatus.FAIL if i in failed else CheckStatus.PASS, evidence="x" if i in failed else None) for i in c.mandatory_ids]
    cs += [Check(check_id=i, status=CheckStatus.FAIL, evidence="x") for i in failed if c.check(i) and i not in c.mandatory_ids]
    return evaluate(c, cs, [], executed_ok=True)[0].status.value


def main():
    out = {}
    for k, v in json.loads((HERE / "expected_verdicts.json").read_text()).items():
        failed = sorted({cid for code in v["codes"] if (cid := check_for(code))})
        out[k] = {"path": v["path"], "profile": "sklearn-tabular", "split": "dev", "framework": "scikit-learn", "expected_checks": failed,
                  "expected_status": expected_status("sklearn-tabular", set(failed)), "baseline_verdict": v["verdict"]}
    for k, v in json.loads((HERE / "expected_adversarial.json").read_text()).items():
        out[k] = {"path": v["path"], "profile": "sklearn-tabular", "split": "dev", "framework": "scikit-learn", "expected_checks": ADVERSARIAL[k],
                  "expected_status": expected_status("sklearn-tabular", set(ADVERSARIAL[k])), "note": v["note"]}
    multi = HERE / "multi_expected.json"
    if multi.exists():
        for k, v in json.loads(multi.read_text()).items():
            out[k] = {**v, "split": "holdout", "expected_status": expected_status(v["profile"], set(v["expected_checks"]))}
    (HERE / "annotations.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(out)} projets annotés -> eval/annotations.json")


if __name__ == "__main__":
    main()
