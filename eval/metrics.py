"""Métriques d'évaluation (fonctions pures, testées) : Validator, Judge, Repair.

Validator : rappel et précision des problèmes (au niveau des contrôles en échec), taux de fausses alertes, fiabilité des décisions,
et le FAUX ACCEPTÉ (projet non conforme accepté) = la mesure la plus grave.
"""
from __future__ import annotations

from collections import defaultdict


def _div(a, b):
    return round(a / b, 4) if b else None


def score_validator(results: list[dict]) -> dict:
    """results : [{name, split, expected_status, predicted_status, expected_checks:[...], predicted_checks:[...]}]"""
    def block(rs):
        tp = fp = fn = 0
        for r in rs:
            e, p = set(r["expected_checks"]), set(r["predicted_checks"])
            tp, fp, fn = tp + len(e & p), fp + len(p - e), fn + len(e - p)
        exp_ok = [r for r in rs if r["expected_status"] == "ACCEPTED"]
        exp_bad = [r for r in rs if r["expected_status"] != "ACCEPTED"]
        false_accept = [r["name"] for r in exp_bad if r["predicted_status"] == "ACCEPTED"]
        false_alarm = [r["name"] for r in exp_ok if r["predicted_status"] != "ACCEPTED"]
        return {"n": len(rs),
                "issue_recall": _div(tp, tp + fn), "issue_precision": _div(tp, tp + fp),
                "false_alarm_rate": _div(len(false_alarm), len(exp_ok)),            # projet conforme signalé à tort
                "decision_reliability": _div(sum(r["expected_status"] == r["predicted_status"] for r in rs), len(rs)),
                "false_accepts": false_accept, "false_accept_rate": _div(len(false_accept), len(exp_bad)),
                "false_rejects": [r["name"] for r in exp_ok if r["predicted_status"] == "REJECTED"],
                "needs_review_rate": _div(sum(r["predicted_status"] == "NEEDS_REVIEW" for r in rs), len(rs)),
                "false_alarms": false_alarm}
    by = defaultdict(list)
    for r in results:
        by[r.get("split", "dev")].append(r)
    return {"all": block(results), **{f"split:{k}": block(v) for k, v in sorted(by.items())}}


def score_judge(results: list[dict]) -> dict:
    """results : [{name, label: good|bad, verdict, failed_closed:bool}] — ACCEPT d'une mauvaise réparation = faux accepté (le plus grave)."""
    good = [r for r in results if r["label"] == "good"]
    bad = [r for r in results if r["label"] == "bad"]
    fa = [r["name"] for r in bad if r["verdict"] == "ACCEPT"]
    fr = [r["name"] for r in good if r["verdict"] != "ACCEPT"]
    ok = sum((r["label"] == "good") == (r["verdict"] == "ACCEPT") for r in results)
    return {"n": len(results), "accuracy": _div(ok, len(results)), "false_accepts": fa, "false_accept_rate": _div(len(fa), len(bad)),
            "false_rejects": fr, "false_reject_rate": _div(len(fr), len(good)), "failed_closed": sum(bool(r.get("failed_closed")) for r in results)}


def score_repair(results: list[dict]) -> dict:
    """results : [{name, repair_status, attempts, revalidated_status|None, regressions:int}] — taux de réparation = revalidé ACCEPTED."""
    n = len(results)
    repaired = [r for r in results if r.get("revalidated_status") == "ACCEPTED"]
    att = [r["attempts"] for r in results if r.get("attempts")]
    return {"n": n, "repair_rate": _div(len(repaired), n), "mean_attempts": round(sum(att) / len(att), 2) if att else None,
            "regressions_introduced": sum(r.get("regressions", 0) for r in results),
            "escalated": [r["name"] for r in results if r["repair_status"] in ("ESCALATED", "FAILED")]}
