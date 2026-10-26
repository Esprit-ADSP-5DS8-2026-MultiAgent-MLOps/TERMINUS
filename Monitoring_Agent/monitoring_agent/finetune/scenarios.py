"""
Générateur de scénarios : on CASSE la production d'une façon connue, puis le VRAI moteur d'analyse
(Analyzer, le même qu'en production) calcule les signaux. L'étiquette vient de deux sources
indépendantes qui doivent être d'accord :

    ce qu'on a injecté (la vérité du « monde »)  ×  ce que la procédure conclut des signaux (l'oracle)

Si elles divergent (ex. un shift trop faible pour être détecté), le scénario est JETÉ : on n'apprend
au modèle que des cas où la bonne réponse est à la fois vraie et déductible des faits qu'il voit.

Injections (toutes sur la fenêtre courante ; parfois en rampe sur les 2 fenêtres d'avant) :
- calm            : rien (le moteur peut lever de faux petits signaux : c'est voulu)
- covariate       : échantillonnage biaisé sur 1-3 features -> P(x) change, P(y|x) non (souvent bénin)
- covariate_hard  : on sur-échantillonne les régions où le modèle se trompe / hésite -> la perf baisse
- concept         : on change la relation x -> y (labels inversés / cible décalée dans une région)
- missing, unseen, type_mismatch, unit_bug, missing_column : pannes du pipeline de données
- mixed           : covariate + une panne de données (la panne doit l'emporter)
Labels en retard : pour les shifts, la fenêtre courante n'a parfois pas encore de labels.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..agent import prompts
from ..agent.context import Case, case_facts
from ..agent.graph import build_messages
from ..agent.rules import diagnose_by_rules, procedure
from ..analysis import Analyzer
from ..config import AnalyzerConfig
from .bases import LOADERS, PRED, PROBA, TARGET, Prepared, prepare

ACCEPT: dict[str, set[str]] = {
    "calm": {"no_issue", "benign_drift"},
    "covariate": {"benign_drift", "covariate_shift"},
    "covariate_hard": {"covariate_shift"},
    "concept": {"concept_drift"},
    "missing": {"data_quality"}, "unseen": {"data_quality"}, "type_mismatch": {"data_quality"},
    "unit_bug": {"data_quality"}, "missing_column": {"data_quality"}, "mixed": {"data_quality"},
}
WEIGHTS = {"calm": 0.07, "covariate": 0.19, "covariate_hard": 0.22, "concept": 0.18, "missing": 0.06,
           "unseen": 0.05, "type_mismatch": 0.04, "unit_bug": 0.05, "missing_column": 0.02, "mixed": 0.12}
GARBAGE = ["N/A", "error", "#VALUE!", "null", "-", "?"]
NEW_CATS = ["UNKNOWN", "n/a", "autre", "INVALID_CODE", "v2_{c}", "{c}_NEW"]


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


@cache
def _scored_pool(name: str) -> pd.DataFrame:
    return prepare(name).model.score(prepare(name).pool)


@cache
def _analyzer(name: str, window: int) -> Analyzer:
    p = prepare(name)
    return Analyzer(p.reference, p.base.mapping(), AnalyzerConfig(window_size=window))


# ---------------------------------------------------------------------------
# Injections
# ---------------------------------------------------------------------------


def _numeric(p: Prepared) -> list[str]:
    return [f for f in p.base.features if f not in p.base.categorical]


def _feature_weights(pool: pd.DataFrame, p: Prepared, rng, intensity: float) -> tuple[np.ndarray, dict]:
    """Covariate shift réaliste : on ne modifie pas les valeurs, on change leur FRÉQUENCE."""
    feats = [str(f) for f in rng.choice(p.base.features, size=min(len(p.base.features), int(rng.integers(1, 4))),
                                        replace=False)]
    w = np.ones(len(pool))
    detail = {}
    for f in feats:
        if f in p.base.categorical:
            cats = pool[f].astype(str)
            target = str(rng.choice(cats.unique()))
            w *= np.where(cats == target, 1 + 8 * intensity, 1.0)
            detail[f] = f"catégorie « {target} » sur-représentée"
        else:
            z = pool[f].rank(pct=True).fillna(0.5).to_numpy()
            sign = rng.choice([-1, 1])
            w *= np.exp(sign * 5 * intensity * (z - 0.5))
            detail[f] = "queue haute" if sign > 0 else "queue basse"
    return w, detail


def _hard_weights(pool: pd.DataFrame, p: Prepared, intensity: float) -> np.ndarray:
    """Régions où le modèle hésite (classification) ou se trompe (régression)."""
    if p.base.task == "classification":
        margin = np.abs(2 * pool[PROBA].to_numpy() - 1)            # 0 = hésitation totale
        return np.exp(-6 * intensity * margin)
    err = np.abs(pool[PRED].to_numpy() - pool[TARGET].to_numpy())
    return (err / (np.median(err) + 1e-9) + 0.05) ** (1.5 * intensity)


def _sample(pool: pd.DataFrame, n: int, rng, weights: np.ndarray | None = None) -> pd.DataFrame:
    prob = None if weights is None else weights / weights.sum()
    idx = rng.choice(len(pool), size=n, replace=len(pool) < n * 1.2, p=prob)
    return pool.iloc[idx].reset_index(drop=True)


def _concept(df: pd.DataFrame, p: Prepared, rng, intensity: float) -> dict:
    """La relation x -> y change dans une région (ou partout) ; les entrées, elles, ne bougent pas."""
    f = str(rng.choice(p.base.features))
    if f in p.base.categorical:
        val = rng.choice(df[f].astype(str).unique())
        region = (df[f].astype(str) == val).to_numpy()
        where = f"{f} = « {val} »"
    else:
        q = rng.uniform(0.3, 0.7)
        cut = df[f].quantile(q)
        region = (df[f] > cut).to_numpy() if rng.random() < 0.5 else (df[f] <= cut).to_numpy()
        where = f"une plage de {f}"
    if region.mean() < 0.25 or rng.random() < 0.25:
        region, where = np.ones(len(df), bool), "partout"
    if p.base.task == "classification":
        rate = 0.15 + 0.4 * intensity
        flip = region & (rng.random(len(df)) < rate)
        df.loc[flip, TARGET] = 1 - df.loc[flip, TARGET]
        return {"feature": f, "region": where, "flip_rate": round(rate, 2)}
    factor = 1 + rng.choice([-1, 1]) * (0.25 + 0.5 * intensity)
    df[TARGET] = df[TARGET].astype(float)
    df.loc[region, TARGET] = df.loc[region, TARGET] * factor
    return {"feature": f, "region": where, "target_factor": round(factor, 2)}


def _quality(kind: str, df: pd.DataFrame, p: Prepared, rng, intensity: float) -> dict:
    nums, cats = _numeric(p), p.base.categorical
    if kind == "unseen" and not cats:
        kind = "missing"
    if kind in ("type_mismatch", "unit_bug") and not nums:
        kind = "missing"
    if kind == "missing":
        feats = [str(f) for f in rng.choice(p.base.features, size=min(2, len(p.base.features)), replace=False)]
        feats = feats[: int(rng.integers(1, 3))]
        rate = 0.1 + 0.5 * intensity
        for f in feats:
            df[f] = df[f].astype(object) if f in cats else df[f]
            df.loc[rng.random(len(df)) < rate, f] = np.nan
        return {"kind": kind, "features": feats, "rate": round(rate, 2)}
    if kind == "unseen":
        f = str(rng.choice(cats))
        rate = 0.05 + 0.45 * intensity
        new = str(rng.choice(NEW_CATS)).format(c=df[f].astype(str).mode()[0])
        df.loc[rng.random(len(df)) < rate, f] = new
        return {"kind": kind, "features": [f], "rate": round(rate, 2), "value": new}
    if kind == "type_mismatch":
        f = str(rng.choice(nums))
        rate = 0.02 + 0.3 * intensity
        df[f] = df[f].astype(object)
        mask = rng.random(len(df)) < rate
        df.loc[mask, f] = rng.choice(GARBAGE, size=int(mask.sum()))
        return {"kind": kind, "features": [f], "rate": round(rate, 2)}
    if kind == "unit_bug":
        f = str(rng.choice(nums))
        rate = 0.4 + 0.6 * intensity
        col = pd.to_numeric(df[f], errors="coerce")
        span = float(col.max() - col.min()) or 1.0
        factor = float(rng.choice([100, 1000]))
        mask = rng.random(len(df)) < rate
        # ×100 sur une feature positive (cm au lieu de m) ; sinon décalage hors de toute plage
        df.loc[mask, f] = col[mask] * factor if col.min() > 0 else col[mask] + factor * span
        return {"kind": kind, "features": [f], "rate": round(rate, 2), "factor": factor}
    if kind == "missing_column":
        f = str(rng.choice(p.base.features))
        df.drop(columns=[f], inplace=True)
        return {"kind": kind, "features": [f]}
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Un scénario
# ---------------------------------------------------------------------------


def make_scenario(name: str, k: int, injection: str | None = None) -> dict:
    """Scénario n°k de la base `name` (déterministe : même (name, k) -> même scénario)."""
    rng = np.random.default_rng(_seed(name, k))
    p = prepare(name)
    pool = _scored_pool(name)
    injection = injection or rng.choice(list(WEIGHTS), p=np.array(list(WEIGHTS.values())) / sum(WEIGHTS.values()))
    sizes = [s for s in (500, 1000, 2000, 3000) if s <= len(pool) / 1.2] or [len(pool) // 2]
    window = int(rng.choice(sizes))
    n_hist = int(rng.integers(3, 9))
    intensity = float(rng.uniform(0.25, 1.0))
    ramp = injection != "calm" and rng.random() < 0.3
    delayed = (injection == "covariate" and rng.random() < 0.3) or (injection == "covariate_hard" and rng.random() < 0.5)

    def window_df(level: float) -> tuple[pd.DataFrame, dict]:
        detail: dict = {}
        weights = None
        if level > 0 and injection in ("covariate", "mixed"):
            weights, detail["shift"] = _feature_weights(pool, p, rng, level)
        elif level > 0 and injection == "covariate_hard":
            weights = _hard_weights(pool, p, level)
            fw, detail["shift"] = _feature_weights(pool, p, rng, level * 0.5)
            weights = weights * fw
        df = _sample(pool, window, rng, weights)[p.base.features + [TARGET]].copy()
        if level > 0 and injection == "concept":
            detail["concept"] = _concept(df, p, rng, level)
        if level > 0 and injection not in ("calm", "covariate", "covariate_hard", "concept"):
            kind = str(rng.choice(["missing", "unseen", "type_mismatch", "unit_bug"])) if injection == "mixed" else injection
            detail["quality"] = _quality(kind, df, p, rng, level)
        scored = p.model.score(df)
        return scored, detail

    levels = [0.0] * n_hist + [intensity]
    if ramp:
        levels[-3:-1] = [intensity * 0.35, intensity * 0.7]
    an = _analyzer(name, window)
    reports, detail = [], {}
    for i, lvl in enumerate(levels):
        df, d = window_df(lvl)
        if i == len(levels) - 1:
            detail = d
            if delayed:
                df[TARGET] = np.nan
        reports.append(an.analyze_window(df, i))

    now = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(hours=int(rng.integers(0, 2000)))
    decisions = []
    if rng.random() < 0.25 and n_hist >= 3:
        prev = rng.choice(["increase_monitoring", "no_action", "open_incident"])
        decisions = [{"id": f"D{n_hist - 2:04d}-prev", "window_index": n_hist - 2, "action": prev, "status": "executed",
                      "mode": "auto", "created_at": (now - timedelta(hours=3)).isoformat()}]
    case = Case(project=name, mapping=p.base.mapping(), history=reports, decisions=decisions,
                model_uri=f"models:/{name}/1", now=now)
    cause, action = procedure(case)
    facts = case_facts(case)
    return {
        "id": f"{name}-{k:05d}", "base": name, "split": p.base.split, "task": p.base.task,
        "injection": str(injection), "accepted": cause in ACCEPT[str(injection)],
        "params": {"window": window, "history": n_hist, "intensity": round(intensity, 2), "ramp": ramp,
                   "labels_delayed": delayed, **detail},
        "oracle": {"root_cause": cause, "action": action},
        "triggered": reports[-1].trigger_agent, "signals": sorted(case.signal_kinds()),
        "features": p.base.features, "facts": facts,
        "messages": build_messages(prompts.DIAGNOSER, case, facts),
        "rules": json.loads(diagnose_by_rules(case).model_dump_json()),
        "prompt_version": prompts.PROMPT_VERSION,
    }


def _generate_base(parts: Path, name: str, target: int, max_attempts_factor: float) -> tuple[str, Counter, str]:
    """Une base, dans son propre processus. Fichiers propres à la base : reprise possible à tout moment."""
    kept_path, seen_path = parts / f"{name}.jsonl", parts / f"{name}.seen"
    try:
        prepare(name)
    except FileNotFoundError as exc:
        return name, Counter(), f"ignorée ({exc})"
    have = sum(1 for _ in kept_path.open(encoding="utf-8")) if kept_path.exists() else 0
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()
    stats: Counter = Counter()
    k = tried = 0
    with kept_path.open("a", encoding="utf-8") as kept, seen_path.open("a") as seen_f:
        while have < target and tried < target * max_attempts_factor:
            sid = f"{name}-{k:05d}"
            k += 1
            if sid in seen:
                continue
            tried += 1
            sc = make_scenario(name, k - 1)
            stats[(sc["injection"], sc["accepted"])] += 1
            if sc["accepted"]:
                kept.write(json.dumps(sc, ensure_ascii=False) + "\n")
                have += 1
            seen_f.write(sid + "\n")
    return name, stats, f"{have} scénarios ({tried} nouveaux essais)"


def generate(out: Path, per_base: dict[str, int], max_attempts_factor: float = 3.0, workers: int | None = None,
             progress: Callable[[str], None] = print) -> Counter:
    """Scénarios ACCEPTÉS jusqu'à `per_base[base]` par base, une base par processus, puis fusion dans `out`.
    Reprenable : relancer la commande continue là où elle s'était arrêtée."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    unknown = set(per_base) - set(LOADERS)
    if unknown:
        raise KeyError(f"bases inconnues : {sorted(unknown)}")
    parts = out.parent / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(len(per_base), (os.cpu_count() or 2) - 1))
    stats: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_generate_base, parts, n, t, max_attempts_factor) for n, t in per_base.items()]
        for fut in as_completed(futures):
            name, st, msg = fut.result()
            progress(f"  {name:<17} {msg}")
            stats.update({(name, inj, ok): v for (inj, ok), v in st.items()})
    with out.open("w", encoding="utf-8") as f:
        for name in per_base:
            part = parts / f"{name}.jsonl"
            if part.exists():
                f.write(part.read_text(encoding="utf-8"))
    return stats
