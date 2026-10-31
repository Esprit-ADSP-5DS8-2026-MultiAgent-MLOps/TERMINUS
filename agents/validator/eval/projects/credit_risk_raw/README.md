# Credit Risk Prediction Model

An end-to-end machine learning pipeline that predicts whether a loan applicant
is a good or bad credit risk, using the classic **UCI German Credit dataset**
(1,000 real loan applicants, 20 attributes each).

This project was built to be genuinely understood and explained, not just run.
Every design decision below is one you should be able to talk through in an
interview.

## Problem statement

A bank or lender wants to decide, at the point of application, whether a
borrower is likely to repay a loan. Getting this wrong in either direction is
costly:
- **False negative** (predicting "good" for someone who defaults) → the bank loses money on the loan.
- **False positive** (predicting "bad" for someone who would have repaid) → the bank loses a good customer.

This asymmetry is why the project reports **precision, recall, and ROC-AUC**
side by side rather than just accuracy — accuracy alone can hide which kind
of mistake a model is making.

## Dataset

- Source: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)
- 1,000 applicants, 20 features (checking account status, credit history,
  loan purpose, loan amount, savings, employment duration, age, housing,
  etc.) plus a binary target: **1 = good credit risk, 0 = bad credit risk**.
- Class balance: ~70% good / ~30% bad — mildly imbalanced, which is why
  `class_weight="balanced"` is used for logistic regression and random
  forest.

## Pipeline

```
data/german_credit_raw.csv
        │
        ▼
src/data_prep.py      → cleans data, engineers features, saves clean CSV
        │
        ▼
src/train.py           → trains 3 models, picks the best by ROC-AUC, saves it
        │
        ▼
src/explain.py         → permutation importance: which features matter most
        │
        ▼
src/predict.py         → scores a new applicant using the saved model
```

### 1. Data preparation (`src/data_prep.py`)
- Separates numeric fields (age, loan amount, duration, etc.) from
  categorical fields (checking account status, purpose, housing, etc.)
- Engineers three extra features:
  - `amount_per_month` — loan amount divided by duration, as a rough proxy
    for monthly repayment burden (two loans of equal size but different
    terms carry different risk).
  - `age_bucket` — age binned into ranges, since credit risk often isn't
    linear in age.
  - `is_long_duration` — flags loans in the top quartile of duration.

### 2. Modeling (`src/train.py`)
Three models are trained and compared on an 80/20 stratified train/test
split:

| Model | Why it's included |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients are directly explainable to a credit committee. |
| Random Forest | Captures non-linear interactions (e.g. "high loan amount AND short employment history" matters more than either alone). |
| XGBoost | Typically the strongest performer on tabular data; industry-standard for credit scoring. |

All three share a common preprocessing pipeline (`StandardScaler` for
numeric features, `OneHotEncoder` for categorical features) wrapped in a
single `sklearn.Pipeline`, so there's no risk of preprocessing the train and
test sets differently.

The model with the **best ROC-AUC on the test set** is automatically
selected and saved to `models/best_model.pkl`.

### 3. Results (from the last run — yours may vary slightly by random seed)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.762 | 0.738 | 0.830 | 0.664 |
| Random Forest | **0.777** | 0.783 | 0.837 | 0.736 |
| XGBoost | 0.775 | **0.826** | 0.804 | **0.850** |

**Random Forest** was selected as the best model by ROC-AUC. Note the
trade-off, though: XGBoost has meaningfully higher recall (catches more bad
credit risks) at a small cost to precision — in a real deployment, which
model you'd actually ship depends on whether the business cares more about
not missing defaulters (favor recall) or not rejecting good customers
(favor precision). This is exactly the kind of trade-off worth raising in
an interview.

See `outputs/roc_curves.png` for the visual comparison.

### 4. Explainability (`src/explain.py`)
Uses **permutation importance** (shuffle one feature, see how much ROC-AUC
drops) rather than relying only on a single model's built-in importances,
because it works identically across all three model types and measures
impact on the actual evaluation metric, not just split quality.

Top drivers of predicted risk, in order:
1. **status** (checking account status) — by far the strongest signal
2. **savings** — savings account balance
3. **amount** — loan amount
4. **credit_history** — prior repayment behavior
5. **duration** — loan term length

This lines up with domain intuition: liquidity (checking/savings) and track
record (credit history) dominate over demographic factors like age or job
type — which is also a reassuring fairness signal, since it means the model
isn't leaning heavily on demographic proxies.

See `outputs/feature_importance.png`.

### 5. Prediction (`src/predict.py`)
A standalone script that loads the saved model and scores a new applicant
dict, returning a prediction plus calibrated probability. This is the piece
you'd wrap in an API (e.g. FastAPI) for production use.

## How to run it

```bash
pip install -r requirements.txt

python3 src/data_prep.py    # clean + feature-engineer the data
python3 src/train.py        # train & compare 3 models, save the best
python3 src/explain.py      # generate feature importance chart
python3 src/predict.py      # score an example applicant
```

## Project structure

```
credit-risk-model/
├── data/
│   ├── german_credit_raw.csv       # original dataset
│   └── german_credit_clean.csv     # cleaned + feature-engineered
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── explain.py
│   └── predict.py
├── models/
│   ├── best_model.pkl              # winning pipeline (preprocessing + model)
│   └── all_models.pkl              # all 3 trained pipelines
├── outputs/
│   ├── metrics.json                # full metrics for all models
│   ├── roc_curves.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Honest limitations (worth mentioning in an interview)

- **Dataset size**: 1,000 rows is small by modern standards — results will
  have real variance, and a production model would need far more data.
- **Dataset age/context**: this is a well-known academic dataset from German
  banking in the 1990s; a real deployment would need current, jurisdiction-
  specific data and a fairness/bias audit (protected attributes like sex
  and age are present in the raw data and should be scrutinized carefully
  before any real use).
- **No hyperparameter tuning**: models use reasonable defaults, not a full
  grid/Bayesian search — a natural next step (`GridSearchCV` or `Optuna`).
- **No cross-validation**: a single train/test split is used for simplicity;
  k-fold CV would give more robust performance estimates.
- **Threshold**: predictions use the default 0.5 probability threshold;
  in practice you'd tune this based on the business cost of false positives
  vs. false negatives (see the precision/recall trade-off above).

## Possible extensions

- Add SHAP values for per-applicant explanations ("why was *this* applicant*
  flagged as risky?"), useful for regulatory explainability (e.g. adverse
  action notices).
- Wrap `predict.py` in a FastAPI endpoint for real-time scoring.
- Add hyperparameter tuning and k-fold cross-validation.
- Add a fairness audit across protected attributes (sex, age, foreign worker
  status) using a library like `fairlearn`.
