"""
explain.py
----------
Explainability for the best model: which features drive risk predictions.

Uses permutation importance (model-agnostic, works for any of the three
pipelines) and, for the tree-based models, native feature importances.
Saves a bar chart of the top drivers to outputs/feature_importance.png.
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data_prep import CLEAN_PATH, ROOT
from train import RANDOM_STATE, load_data

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
OUTPUT_PATH = os.path.join(ROOT, "outputs", "feature_importance.png")


def run(top_n: int = 15):
    X, y = load_data(CLEAN_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )

    importances = pd.Series(result.importances_mean, index=X_test.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled\n(higher = more important)")
    ax.set_title(f"Top {top_n} Features Driving Credit Risk Predictions")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved feature importance chart to {OUTPUT_PATH}")

    print("\nTop features:")
    for feat, score in importances.sort_values(ascending=False).items():
        print(f"  {feat:30s} {score:.4f}")

    return importances


if __name__ == "__main__":
    run()
