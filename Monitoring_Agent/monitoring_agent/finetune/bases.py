"""
Bases du dataset de fine-tuning : des jeux SAINS sur lesquels on entraîne un modèle « déployé ».

Chaque base est découpée une fois pour toutes (graine fixe) :
    40 % entraînement du modèle substitut | 30 % référence (scorée) | 30 % réservoir de production

Le découpage train / valid / test se fait PAR BASE (jamais par exemple) : le modèle fine-tuné est évalué
sur des jeux qu'il n'a jamais vus (car_loan, car_price), sinon on mesurerait de la mémorisation.

Choix des bases (tous dans data/) :
- variété de tâches : classification binaire (5) et régression (3) ;
- variété de features : numériques, catégorielles, booléennes, de 7 à 19 colonnes ;
- RAC1P (origine ethnique) et `duration` (fuite de la cible, cf. doc UCI) sont retirés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from ..datasets import DATASETS, find_file
from ..schemas import ColumnMapping

Split = Literal["train", "valid", "test"]
TARGET, PRED, PROBA = "y", "y_pred", "y_proba"


@dataclass
class Base:
    name: str
    split: Split
    task: Literal["classification", "regression"]
    frame: pd.DataFrame                      # features + colonne "y"
    features: list[str]
    description: str
    categorical: list[str] = field(default_factory=list)

    @property
    def metric(self) -> str:
        return "roc_auc" if self.task == "classification" else "mae"

    def mapping(self, features: list[str] | None = None) -> ColumnMapping:
        return ColumnMapping(task=self.task, features=features or self.features, target=TARGET, prediction=PRED,
                             prediction_proba=PROBA if self.task == "classification" else None,
                             primary_metric=self.metric)


def _read(name: str, **kw) -> pd.DataFrame:
    path = find_file(name)
    return pd.read_parquet(path) if path.suffix in {".pq", ".parquet"} else pd.read_csv(path, **kw)


def _clean(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, list[str]]:
    out = df[features].copy()
    cats = []
    for c in features:
        if out[c].dtype == bool or str(out[c].dtype) in ("object", "category"):
            out[c] = out[c].astype(str)
            cats.append(c)
    out[TARGET] = df[target].to_numpy()
    return out.reset_index(drop=True), cats


def _synthetic_sample() -> Base:
    df = _read("synthetic_sample_reference.csv")
    feats = ["distance_from_office", "salary_range", "gas_price_per_litre", "public_transportation_cost",
             "wfh_prev_workday", "workday", "tenure"]
    f, cats = _clean(df, feats, "work_home_actual")
    return Base("synthetic_sample", "train", "classification", f, feats, "télétravail prévu (NannyML)", cats)


def _employment() -> Base:
    df = _read("employment_MA_reference.pq")
    feats = [c for c in df.columns if c not in ("id", "RAC1P", "employed", "year", "prediction", "predicted_probability")]
    f, cats = _clean(df, feats, "employed")
    return Base("employment_MA", "train", "classification", f, feats, "emploi, recensement US (Folktables)", cats)


def _bank() -> Base:
    df = _read("bank-additional-full.csv", sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]   # « emp.var.rate » casserait les clés de faits
    df["y"] = (df["y"] == "yes").astype(int)
    feats = [c for c in df.columns if c not in ("y", "duration")]
    f, cats = _clean(df, feats, "y")
    return Base("bank", "train", "classification", f, feats, "souscription d'un dépôt à terme (UCI)", cats)


def _bike() -> Base:
    df = _read("hour.csv")
    df["season"] = df["season"].map({1: "spring", 2: "summer", 3: "fall", 4: "winter"})
    df["weathersit"] = df["weathersit"].map({1: "clear", 2: "mist", 3: "light_rain", 4: "heavy_rain"})
    feats = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
             "windspeed"]
    f, cats = _clean(df, feats, "cnt")
    return Base("bike", "train", "regression", f, feats, "locations de vélos par heure (UCI)", cats)


def _california() -> Base:
    df = _read("california_housing_reference.csv")
    feats = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    f, cats = _clean(df, feats, "clf_target")
    return Base("california", "train", "classification", f, feats, "prix immobilier élevé ou non (Californie)", cats)


def _california_reg() -> Base:
    df = _read("california_housing_reference.csv")
    feats = ["HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    f, cats = _clean(df, feats, "MedInc")
    return Base("california_reg", "train", "regression", f, feats, "revenu médian d'un quartier (Californie)", cats)


def _mc() -> Base:
    df = _read("mc_reference.csv")
    df["y"] = (df["y_true"] == "prepaid_card").astype(int)
    feats = ["acq_channel", "app_behavioral_score", "requested_credit_limit", "app_channel", "credit_bureau_score",
             "stated_income", "is_customer"]
    f, cats = _clean(df, feats, "y")
    return Base("mc_prepaid", "valid", "classification", f, feats, "carte prépayée ou non (NannyML multiclasse)", cats)


def _car_loan() -> Base:
    spec = DATASETS["car_loan"]
    f, cats = _clean(_read(spec.reference), spec.mapping.features, spec.mapping.target)
    return Base("car_loan", "test", "classification", f, spec.mapping.features, spec.description, cats)


def _car_price() -> Base:
    spec = DATASETS["car_price"]
    f, cats = _clean(_read(spec.reference), spec.mapping.features, spec.mapping.target)
    return Base("car_price", "test", "regression", f, spec.mapping.features, spec.description, cats)


LOADERS = {"synthetic_sample": _synthetic_sample, "employment_MA": _employment, "bank": _bank, "bike": _bike,
           "california": _california, "california_reg": _california_reg, "mc_prepaid": _mc,
           "car_loan": _car_loan, "car_price": _car_price}


@cache
def load_base(name: str) -> Base:
    return LOADERS[name]()


def available() -> dict[str, str]:
    """Bases présentes sur cette machine (les autres sont ignorées, avec la raison)."""
    out = {}
    for name, fn in LOADERS.items():
        try:
            b = fn()
            out[name] = f"{b.split:5} {b.task:14} {len(b.frame):>6} lignes, {len(b.features):>2} features — {b.description}"
        except FileNotFoundError as exc:
            out[name] = f"absente ({exc})"
    return out


# ---------------------------------------------------------------------------
# Le modèle « déployé » : un substitut entraîné sur la base, qui score la production
# ---------------------------------------------------------------------------


class Deployed:
    """HistGradientBoosting : gère nativement valeurs manquantes et catégories (inconnues -> manquantes),
    comme un modèle en production qui reçoit des données abîmées sans planter."""

    def __init__(self, base: Base, seed: int = 0):
        self.base = base
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                      encoded_missing_value=np.nan)
        cat_mask = [c in base.categorical for c in base.features]
        cls = HistGradientBoostingClassifier if base.task == "classification" else HistGradientBoostingRegressor
        self.model = cls(max_iter=200, learning_rate=0.08, categorical_features=cat_mask or None, random_state=seed)
        self.seed = seed

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        b = self.base
        x = pd.DataFrame(index=df.index)
        for c in b.features:
            if c not in df.columns:
                x[c] = np.nan                               # colonne disparue : tout manquant
            elif c in b.categorical:
                x[c] = df[c].astype(object).where(df[c].notna(), None)
            else:
                x[c] = pd.to_numeric(df[c], errors="coerce")  # « N/A », « error »... -> manquant
        if b.categorical:
            x[b.categorical] = self.encoder.transform(x[b.categorical].astype(object).fillna("__nan__"))
        return x.astype(float)

    def fit(self, df: pd.DataFrame) -> Deployed:
        b = self.base
        if b.categorical:
            self.encoder.fit(df[b.categorical].astype(object).fillna("__nan__"))
        self.model.fit(self._x(df), df[TARGET])
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        x = self._x(df)
        if self.base.task == "classification":
            proba = self.model.predict_proba(x)[:, 1]
            out[PROBA] = proba
            out[PRED] = (proba >= 0.5).astype(int)
        else:
            out[PRED] = self.model.predict(x)
        return out


@dataclass
class Prepared:
    base: Base
    model: Deployed
    reference: pd.DataFrame        # scorée, avec labels
    pool: pd.DataFrame             # production « saine », non scorée (on la casse avant de la scorer)


@cache
def prepare(name: str, seed: int = 0) -> Prepared:
    base = load_base(name)
    df = base.frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    a, b = int(0.4 * n), int(0.7 * n)
    model = Deployed(base, seed).fit(df.iloc[:a])
    return Prepared(base, model, model.score(df.iloc[a:b].reset_index(drop=True)), df.iloc[b:].reset_index(drop=True))
