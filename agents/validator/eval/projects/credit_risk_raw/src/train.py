"""
train.py
--------
Trains and compares three models for predicting credit risk:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest
  3. XGBoost

Saves the best-performing model (by ROC-AUC on the held-out test set)
to models/best_model.pkl, along with the preprocessing pipeline.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "german_credit_clean.csv")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def run():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X)
    models = get_models()

    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(
            f"{name:22s} | AUC: {metrics['roc_auc']:.3f} | "
            f"F1: {metrics['f1']:.3f} | Recall: {metrics['recall']:.3f} | "
            f"Precision: {metrics['precision']:.3f}"
        )

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(fitted_pipelines, f"{MODEL_DIR}/all_models.pkl")

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(
            {"best_model": best_name, "results": results}, f, indent=2, default=str
        )

    # Save a ROC curve comparison plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve Comparison — Credit Risk Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150)
    print(f"Saved ROC curve comparison to {OUTPUT_DIR}/roc_curves.png")

    return results, best_name, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run()
