"""
Jeux de données de monitoring (dossier data/, ignoré par git).

Chaque jeu = une référence (données d'entraînement, avec labels) + une période de production
(features + prédictions) + les labels de production, livrés à part comme en vrai.

Les fichiers sont retrouvés par leur NOM n'importe où sous data/ (les noms de dossiers
contiennent des espaces et des accents : on ne reconstruit jamais les chemins à la main).
Dossier : variable MONITORING_DATA_DIR, sinon Monitoring_Agent/data/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from .schemas import ColumnMapping

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # Monitoring_Agent/


def data_dir() -> Path:
    return Path(os.environ.get("MONITORING_DATA_DIR", PACKAGE_ROOT / "data")).expanduser()


@cache
def find_file(name: str) -> Path:
    root = data_dir()
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} introuvable sous {root} (définir MONITORING_DATA_DIR ?)")
    return hits[0]


def _read(name: str) -> pd.DataFrame:
    path = find_file(name)
    return pd.read_parquet(path) if path.suffix in {".pq", ".parquet"} else pd.read_csv(path)


@dataclass(frozen=True)
class DatasetSpec:
    reference: str
    analysis: str
    targets: str | None
    mapping: ColumnMapping
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "car_loan": DatasetSpec(
        reference="synthetic_car_loan_reference.csv",
        analysis="synthetic_car_loan_analysis.csv",
        targets="synthetic_car_loan_analysis_target.csv",
        description="NannyML Car Loan : classification binaire (remboursement d'un crédit auto)",
        mapping=ColumnMapping(
            task="classification", target="repaid", prediction="y_pred", prediction_proba="y_pred_proba",
            timestamp="timestamp", id="id", primary_metric="roc_auc",
            features=["car_value", "salary_range", "debt_to_income_ratio", "loan_length",
                      "repaid_loan_on_prev_car", "size_of_downpayment", "driver_tenure"],
        ),
    ),
    "car_loan_dq": DatasetSpec(
        reference="synthetic_car_loan_dq_reference.csv",
        analysis="synthetic_car_loan_dq_analysis.csv",
        targets="synthetic_car_loan_analysis_target.csv",   # mêmes labels que car_loan (ids alignés)
        description="Car Loan avec problèmes de qualité : valeurs manquantes, catégories inconnues",
        mapping=ColumnMapping(
            task="classification", target="repaid", prediction="y_pred", prediction_proba="y_pred_proba",
            timestamp="timestamp", id="id", primary_metric="roc_auc",
            features=["car_value", "salary_range", "debt_to_income_ratio", "loan_length",
                      "repaid_loan_on_prev_car", "size_of_downpayment", "driver_tenure"],
        ),
    ),
    "car_price": DatasetSpec(
        reference="regression_synthetic_reference.csv",
        analysis="regression_synthetic_analysis.csv",
        targets="regression_synthetic_analysis_targets.csv",
        description="NannyML : régression du prix de voitures d'occasion",
        mapping=ColumnMapping(
            task="regression", target="y_true", prediction="y_pred", timestamp="timestamp", id="id",
            primary_metric="mae",
            features=["car_age", "km_driven", "price_new", "accident_count", "door_count", "fuel", "transmission"],
        ),
    ),
}


def load(name: str, with_labels: bool = True, label_delay_windows: int = 0,
         window_size: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
    """Renvoie (référence, production, spec).

    label_delay_windows : les N dernières fenêtres n'ont PAS encore leurs labels
    (simule le retard de la vérité terrain ; seule la performance estimée y est disponible).
    """
    spec = DATASETS[name]
    m = spec.mapping
    reference = _read(spec.reference)
    production = _read(spec.analysis)
    if with_labels and spec.targets:
        labels = _read(spec.targets)
        production = production.drop(columns=[m.target], errors="ignore").merge(labels, on=m.id, how="left")
        if label_delay_windows > 0:
            cutoff = max(0, len(production) - label_delay_windows * window_size)
            production.loc[production.index >= cutoff, m.target] = None
    return reference, production, spec
