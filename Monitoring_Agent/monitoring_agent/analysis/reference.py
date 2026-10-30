"""
Profil de référence : tout ce qu'on apprend une fois sur les données d'entraînement
et qui sert ensuite à juger chaque fenêtre de production.

- par feature : valeurs, classes du PSI, catégories connues, min/max, taux de manquants ;
- la distribution des prédictions (pour le prediction drift) ;
- la variabilité naturelle de la métrique principale (pour savoir ce qui est « anormal ») ;
- l'estimateur de performance sans labels (CBPE ou DLE), entraîné sur la référence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import AnalyzerConfig
from ..schemas import ColumnKind, ColumnMapping
from . import stats
from .performance import CBPE, DLE, metric_band, realized_metrics


@dataclass
class ColumnReference:
    name: str
    kind: ColumnKind
    n: int
    missing_rate: float
    # numérique
    values: np.ndarray | None = None
    edges: np.ndarray | None = None
    min: float | None = None
    max: float | None = None
    # catégoriel
    categories: list[str] = field(default_factory=list)
    counts: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        if self.kind == "numeric":
            v = self.values
            return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v))} if len(v) else {}
        total = self.counts.sum()
        top = np.argsort(-self.counts)[:3]
        return {self.categories[i]: float(self.counts[i] / total) for i in top} if total else {}


def infer_kind(series: pd.Series) -> ColumnKind:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    return "numeric"


def build_column_reference(series: pd.Series, bins: int) -> ColumnReference:
    kind = infer_kind(series)
    ref = ColumnReference(name=str(series.name), kind=kind, n=len(series),
                          missing_rate=float(series.isna().mean()))
    if kind == "numeric":
        vals = stats.clean_numeric(series)
        ref.values = vals
        ref.edges = stats.quantile_edges(vals, bins)
        ref.min, ref.max = (float(vals.min()), float(vals.max())) if len(vals) else (None, None)
    else:
        counts = stats.as_category_strings(series).value_counts()
        ref.categories = counts.index.tolist()
        ref.counts = counts.to_numpy(dtype=float)
    return ref


@dataclass
class ReferenceProfile:
    mapping: ColumnMapping
    config: AnalyzerConfig
    n_rows: int
    columns: dict[str, ColumnReference]
    prediction: ColumnReference
    metric_mean: float | None = None
    metric_std: float | None = None
    estimator: CBPE | DLE | None = None
    reference_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, mapping: ColumnMapping, config: AnalyzerConfig | None = None) -> ReferenceProfile:
        config = config or AnalyzerConfig()
        bins = config.drift.psi_bins
        missing = [c for c in mapping.features + [mapping.prediction] if c not in df.columns]
        if missing:
            raise ValueError(f"colonnes absentes de la référence : {missing}")

        columns = {c: build_column_reference(df[c], bins) for c in mapping.features}
        # Prédiction : on suit le score (proba) s'il existe, sinon la classe / la valeur prédite.
        pred_col = mapping.prediction_proba or mapping.prediction
        prediction = build_column_reference(df[pred_col], bins)

        profile = cls(mapping=mapping, config=config, n_rows=len(df), columns=columns, prediction=prediction)
        if mapping.target and mapping.target in df.columns and df[mapping.target].notna().any():
            profile._fit_performance(df[df[mapping.target].notna()])
        return profile

    # -- performance -----------------------------------------------------------

    def _metrics(self, df: pd.DataFrame) -> dict[str, float]:
        m = self.mapping
        proba = df[m.prediction_proba] if m.prediction_proba else None
        return realized_metrics(m.task, df[m.target], df[m.prediction], proba)

    def _fit_performance(self, df: pd.DataFrame) -> None:
        m, cfg = self.mapping, self.config.performance
        self.reference_metrics = self._metrics(df)

        # Variabilité naturelle : la métrique sur des fenêtres de la référence, de même taille
        # que les fenêtres de production (une métrique sur 5 000 lignes fluctue moins que sur 500).
        size = self.config.window_size
        n_chunks = len(df) // size
        if n_chunks >= cfg.min_reference_windows:
            chunks = [df.iloc[i * size:(i + 1) * size] for i in range(n_chunks)]
        else:
            rng = np.random.default_rng(cfg.random_state)
            k = min(size, len(df))
            chunks = [df.iloc[rng.integers(0, len(df), k)] for _ in range(cfg.bootstrap_samples)]
        values = [self._metrics(c).get(m.primary_metric, np.nan) for c in chunks]
        self.metric_mean, self.metric_std = metric_band(values)

        # Estimateur sans labels
        binary = m.task == "classification" and m.prediction_proba and df[m.target].nunique() == 2
        if binary:
            self.estimator = CBPE().fit(df[m.prediction_proba], df[m.target])
        elif m.task == "regression":
            cats = [c for c in m.features if self.columns[c].kind == "categorical"]
            self.estimator = DLE(m.features, cats, cfg.random_state).fit(df, df[m.prediction], df[m.target])
