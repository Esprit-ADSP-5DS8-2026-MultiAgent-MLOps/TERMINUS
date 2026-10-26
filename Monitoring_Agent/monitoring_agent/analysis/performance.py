"""
Performance du modèle : mesurée quand les labels sont là, ESTIMÉE quand ils manquent.

Pourquoi estimer ? En production, la vérité terrain arrive tard (un prêt est remboursé
ou non des mois après la prédiction). Attendre les labels, c'est découvrir la panne trop tard.

- Classification binaire : CBPE (Confidence-Based Performance Estimation).
  Si les probabilités du modèle sont bien calibrées, une prédiction « positif » avec p = 0.8
  est juste 8 fois sur 10. En sommant ces probabilités on obtient une matrice de confusion
  ATTENDUE, donc accuracy, précision, rappel, F1 et ROC AUC attendus, sans aucun label.
- Régression : DLE (Direct Loss Estimation).
  Sur la référence, on apprend un second modèle qui prédit l'ERREUR du premier
  à partir des features. En production, il estime l'erreur moyenne (MAE, RMSE).

Limite à connaître : ces estimations captent la dégradation due au déplacement des données
(covariate shift). Si la relation features -> cible change (concept drift), seule la
performance mesurée avec de vrais labels le verra. Les deux sont donc complémentaires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Performance mesurée (labels disponibles)
# ---------------------------------------------------------------------------


def realized_metrics(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {}
    if task == "regression":
        err = y_true.astype(float) - y_pred.astype(float)
        out = {"mae": float(mean_absolute_error(y_true, y_pred)),
               "rmse": float(np.sqrt(np.mean(err ** 2)))}
        if len(y_true) > 1:
            out["r2"] = float(r2_score(y_true, y_pred))
        return out

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    binary = len(set(np.unique(y_true)) | set(np.unique(y_pred))) <= 2
    avg = "binary" if binary else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if binary and y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba, dtype=float)))
    return out


# ---------------------------------------------------------------------------
# CBPE : classification binaire
# ---------------------------------------------------------------------------


class CBPE:
    def __init__(self) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_proba, y_true) -> CBPE:
        """Calibration sur la référence : apprend P(y=1 | score) réellement observée."""
        self.calibrator.fit(np.asarray(y_proba, dtype=float), np.asarray(y_true, dtype=float))
        return self

    def estimate(self, y_proba, y_pred) -> dict[str, float]:
        score = np.asarray(y_proba, dtype=float)
        y_pred = np.asarray(y_pred).astype(int)
        if len(score) == 0:
            return {}
        p = self.calibrator.predict(score)          # probabilité calibrée que y = 1
        pos = y_pred == 1
        tp, fp = p[pos].sum(), (1 - p[pos]).sum()   # matrice de confusion ATTENDUE
        fn, tn = p[~pos].sum(), (1 - p[~pos]).sum()
        n = len(p)
        return {
            "accuracy": float((tp + tn) / n),
            "precision": float(tp / (tp + fp)) if tp + fp > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn > 0 else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn > 0 else 0.0,
            "roc_auc": expected_roc_auc(score, p),
        }


def expected_roc_auc(score: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC attendue : pour chaque seuil sur le score, TPR et FPR attendus calculés
    avec les probabilités calibrées p, puis aire sous la courbe."""
    order = np.argsort(-score, kind="mergesort")
    s, p = score[order], p[order]
    tp = np.cumsum(p)
    fp = np.cumsum(1 - p)
    last_of_group = np.r_[s[1:] != s[:-1], True]       # ex æquo : un seul point par seuil
    tp, fp = tp[last_of_group], fp[last_of_group]
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# DLE : régression
# ---------------------------------------------------------------------------


class DLE:
    def __init__(self, features: list[str], categorical: list[str], random_state: int = 42):
        self.features, self.categorical = features, categorical
        self.categories: dict[str, list[str]] = {}
        params = {"max_iter": 200, "min_samples_leaf": 50, "random_state": random_state,
                      "categorical_features": "from_dtype"}
        self.abs_model = HistGradientBoostingRegressor(**params)
        self.sq_model = HistGradientBoostingRegressor(**params)

    def _frame(self, df: pd.DataFrame, y_pred) -> pd.DataFrame:
        # Données abîmées (colonne absente, texte dans une colonne numérique) : l'estimation doit
        # continuer — c'est justement là que la qualité des données lève son propre signal.
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            col = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
            if c in self.categorical:
                X[c] = pd.Categorical(col.astype("string"), categories=self.categories[c])
            else:
                X[c] = pd.to_numeric(col, errors="coerce")
        X["__prediction__"] = np.asarray(y_pred, dtype=float)
        return X

    def fit(self, df: pd.DataFrame, y_pred, y_true) -> DLE:
        for c in self.categorical:
            self.categories[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        X = self._frame(df, y_pred)
        err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        self.abs_model.fit(X, np.abs(err))
        self.sq_model.fit(X, err ** 2)
        return self

    def estimate(self, df: pd.DataFrame, y_pred) -> dict[str, float]:
        if len(df) == 0:
            return {}
        X = self._frame(df, y_pred)
        mae = float(np.mean(np.clip(self.abs_model.predict(X), 0, None)))
        rmse = float(np.sqrt(np.mean(np.clip(self.sq_model.predict(X), 0, None))))
        return {"mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Variabilité naturelle sur la référence
# ---------------------------------------------------------------------------


def metric_band(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
