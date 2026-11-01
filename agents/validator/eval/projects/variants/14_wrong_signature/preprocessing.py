import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AGE_BINS = [0, 25, 35, 45, 55, 100]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]


class CreditFeatures(BaseEstimator, TransformerMixin):
    """Feature engineering appris A L'ENTRAINEMENT.

    Le seuil de durée longue est mémorisé au fit : il n'est plus recalculé
    sur le lot reçu à l'inférence (corrige le skew du projet d'origine).
    """

    def fit(self, X, y=None):
        self.long_duration_threshold_ = float(pd.Series(X["duration"]).quantile(0.75))
        return self

    def transform(self, X):
        df = X.copy()
        df["amount_per_month"] = df["amount"] / df["duration"].replace(0, 1)
        df["age_bucket"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        df["is_long_duration"] = (df["duration"] >= self.long_duration_threshold_).astype(int)
        return df


def build_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("features", CreditFeatures()), ("prep", prep), ("model", model)])
