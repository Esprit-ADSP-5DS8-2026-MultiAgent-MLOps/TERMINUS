"""
data_prep.py
------------
Loads the raw German Credit dataset and cleans it into a model-ready form.

Source data: UCI Statlog (German Credit Data), 1000 loan applicants.
Target: credit_risk (1 = good credit / repaid, 0 = bad credit / defaulted)
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "german_credit_raw.csv")
CLEAN_PATH = os.path.join(ROOT, "data", "german_credit_clean.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target: dataset encodes 1 = good, 0 = bad already in this mirror.
    # We rename for clarity and keep it as-is (1 = good / low risk).
    df = df.rename(columns={"credit_risk": "target"})

    # Basic sanity: drop exact duplicate rows if any
    df = df.drop_duplicates()

    # Split columns into numeric vs categorical for downstream encoding
    numeric_cols = [
        "duration",
        "amount",
        "installment_rate",
        "present_residence",
        "age",
        "number_credits",
        "people_liable",
    ]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["target"]]

    # Coerce numerics, fill any stray NaNs with median (defensive; this
    # dataset has no missing values, but real-world data will)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    for c in categorical_cols:
        df[c] = df[c].astype(str).fillna("unknown")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """A few domain-informed features on top of the raw fields."""
    df = df.copy()

    # Loan amount relative to duration -> monthly burden proxy
    df["amount_per_month"] = df["amount"] / df["duration"].replace(0, 1)

    # Age buckets often behave non-linearly in credit risk
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)

    # Is this a long-duration loan (top quartile)?
    df["is_long_duration"] = (df["duration"] >= df["duration"].quantile(0.75)).astype(int)

    return df


def run(raw_path: str = RAW_PATH, out_path: str = CLEAN_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    df.to_csv(out_path, index=False)
    print(f"Cleaned dataset saved to {out_path} — shape: {df.shape}")
    return df


if __name__ == "__main__":
    run()
