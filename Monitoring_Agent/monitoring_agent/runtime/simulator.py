"""
Simulateur de trafic de production : envoie des requêtes à la gateway comme le ferait une
application cliente, puis renvoie les labels avec du retard (comme la vraie vie).

Scénarios (après une phase normale de `warmup` requêtes) :
- normal     : même distribution que l'entraînement            -> rien ne doit se passer
- covariate  : les 3 features numériques les plus variées augmentent (x1.8) -> drift.
               Le label reste celui de la ligne d'origine : pour le modèle, la relation
               features -> cible change aussi. La perf MESURÉE chute, l'ESTIMÉE beaucoup moins.
- quality    : catégories inconnues + valeurs aberrantes        -> problème de qualité, pas de « vrai » drift
- concept    : features inchangées, mais la relation features -> cible change
               (labels inversés sur une partie des cas)        -> AUCUN drift, seule la perf mesurée baisse

Le scénario `concept` montre la limite de l'estimation sans labels (CBPE) : il faut la vérité terrain.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

SCENARIOS = ("normal", "covariate", "quality", "concept")


@dataclass
class SimulationResult:
    sent: int = 0
    ok: int = 0
    errors: int = 0
    labels_sent: int = 0
    seconds: float = 0.0


def _numeric(df: pd.DataFrame) -> list[str]:
    """Features numériques, de la plus variée à la moins variée."""
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    return sorted(cols, key=lambda c: -df[c].nunique())


def perturb(row: dict, scenario: str, rng: random.Random, numeric: list[str], categorical: list[str],
            intensity: float) -> tuple[dict, bool]:
    """Applique le scénario à une ligne. Renvoie (ligne, inverser_le_label)."""
    row = dict(row)
    if scenario == "covariate":
        for col in numeric[:3]:                                   # les 3 features numériques les plus variées
            if row.get(col) is not None:
                row[col] = type(row[col])(row[col] * (1 + 0.8 * intensity))
        return row, False
    if scenario == "quality":
        if categorical and rng.random() < 0.25 * intensity:
            row[rng.choice(categorical)] = "UNKNOWN_" + rng.choice(["A", "B"])
        if numeric and rng.random() < 0.15 * intensity:
            col = rng.choice(numeric)
            if row.get(col) is not None:
                row[col] = type(row[col])(row[col] * 100)
        return row, False
    if scenario == "concept":
        return row, rng.random() < 0.35 * intensity
    return row, False


async def simulate(gateway_url: str, data: pd.DataFrame, target: str, scenario: str = "normal",
                   n: int = 1600, warmup: int = 600, rate: float = 100.0, label_delay: int = 200,
                   seed: int = 7, concurrency: int = 8, ramp: int = 200, final_flush: bool = False,
                   on_progress=None) -> SimulationResult:
    """`ramp` : nombre de requêtes pour passer de 0 à 100 % d'intensité (un drift réel est progressif).
    `final_flush` : à la fin, envoie aussi les derniers labels (la vérité finit toujours par arriver).
    `on_progress(res)` : appelé après chaque lot (barre de progression de la console)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scénario inconnu : {scenario} ({', '.join(SCENARIOS)})")
    rng = random.Random(seed)
    features = data.drop(columns=[target])
    numeric = _numeric(features)
    categorical = [c for c in features.columns if c not in numeric]
    rows = features.to_dict(orient="records")
    labels = data[target].tolist()

    res = SimulationResult()
    pending: deque[tuple[str, float]] = deque()   # labels pas encore « connus »
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    async with httpx.AsyncClient(base_url=gateway_url, timeout=15.0) as client:
        async def send(i: int):
            k = rng.randrange(len(rows))
            row, label = rows[k], labels[k]
            flip = False
            if i >= warmup and scenario != "normal":
                intensity = min(1.0, (i - warmup + 1) / max(ramp, 1))
                row, flip = perturb(row, scenario, rng, numeric, categorical, intensity)
            row = {c: (None if isinstance(v, float) and np.isnan(v) else v) for c, v in row.items()}
            rid = uuid.uuid4().hex
            async with sem:
                try:
                    r = await client.post("/predict", json=row, headers={"X-Request-ID": rid})
                    ok = r.is_success
                except httpx.HTTPError:
                    ok = False
            res.sent += 1
            if ok:
                res.ok += 1
                pending.append((rid, float(1 - label if flip else label)))
            else:
                res.errors += 1

        batch = max(1, int(rate // 10))           # ~10 lots par seconde
        for start in range(0, n, batch):
            tick = time.perf_counter()
            await asyncio.gather(*(send(i) for i in range(start, min(start + batch, n))))
            while len(pending) > label_delay:     # la vérité terrain arrive avec `label_delay` requêtes de retard
                items = [pending.popleft() for _ in range(min(100, len(pending) - label_delay))]
                await client.post("/feedback", json={"items": [{"request_id": r, "label": y} for r, y in items]})
                res.labels_sent += len(items)
            if on_progress:
                on_progress(res)
            await asyncio.sleep(max(0.0, 0.1 - (time.perf_counter() - tick)))

        if (label_delay == 0 or final_flush) and pending:
            items = list(pending)
            await client.post("/feedback", json={"items": [{"request_id": r, "label": y} for r, y in items]})
            res.labels_sent += len(items)

    res.seconds = time.perf_counter() - t0
    return res
