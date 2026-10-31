"""
predict.py
----------
Loads the trained model and scores new loan applicants.

Usage:
    python3 src/predict.py            # runs on a built-in example applicant
"""

import os

import joblib
import pandas as pd

from data_prep import engineer_features
from train import ROOT

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")

# An example applicant, using the same raw field names as the source dataset.
EXAMPLE_APPLICANT = {
    "status": "no checking account",
    "duration": 24,
    "credit_history": "existing credits paid back duly till now",
    "purpose": "car (new)",
    "amount": 3500,
    "savings": "... < 100 DM",
    "employment_duration": "1 <= ... < 4 years",
    "installment_rate": 3,
    "personal_status_sex": "male : single",
    "other_debtors": "none",
    "present_residence": 2,
    "property": "car or other",
    "age": 29,
    "other_installment_plans": "none",
    "housing": "rent",
    "number_credits": 1,
    "job": "skilled employee/official",
    "people_liable": 1,
    "telephone": "yes",
    "foreign_worker": "yes",
}


def score_applicant(applicant: dict, model_path: str = MODEL_PATH) -> dict:
    pipeline = joblib.load(model_path)

    df = pd.DataFrame([applicant])
    df = engineer_features(df)

    proba_good = pipeline.predict_proba(df)[0, 1]
    prediction = pipeline.predict(df)[0]

    return {
        "prediction": "GOOD credit risk (likely to repay)" if prediction == 1
        else "BAD credit risk (elevated default risk)",
        "probability_good": round(float(proba_good), 4),
        "probability_bad": round(float(1 - proba_good), 4),
    }


if __name__ == "__main__":
    result = score_applicant(EXAMPLE_APPLICANT)
    print("Example applicant:")
    for k, v in EXAMPLE_APPLICANT.items():
        print(f"  {k}: {v}")
    print("\nPrediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
