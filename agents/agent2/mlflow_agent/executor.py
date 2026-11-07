"""Executor: runs a validated plan and returns an ExperimentResult.

Security design: project code (train.py etc.) is NEVER executed. We only read the
CSV and train our own bounded scikit-learn models.
"""
from __future__ import annotations

import hashlib
import inspect
import time

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from mlflow_agent.plans import ExperimentPlan
from mlflow_agent.results import ExperimentResult, QualityGate
from mlflow_agent.schemas import AnalysisReport, ResolvedPaths

SEED = 42
MAX_FILE_BYTES = 50_000_000
MAX_ROWS = 200_000
MIN_ROWS = 50
# Quality-gate constants (assumptions to justify in the report, adjust if needed)
MAX_REL_OVERFIT_GAP = 0.25   # (train - validation) / train, in "higher is better" terms
MAX_REL_TEST_DROP = 0.25     # (validation - test) / validation
MINIMIZE = {"rmse", "mae"}
# Model selection by k-fold cross-validation on train+validation. With small data (credit_risk:
# 200 validation rows) one validation split is too noisy: it picked gradient boosting (val 0.812,
# test 0.771) over random forest (val 0.806, test 0.792).
CV_FOLDS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ExecutionError(Exception):
    """A known, reportable problem during execution."""


def _sign(metric: str) -> int:
    return -1 if metric in MINIMIZE else 1


# ---------- data ----------

def load_xy(paths: ResolvedPaths, target: str, warnings: list[str]):
    if paths.dataset.stat().st_size > MAX_FILE_BYTES:
        raise ExecutionError("dataset file too large")
    df = pd.read_csv(paths.dataset)
    if not MIN_ROWS <= len(df) <= MAX_ROWS:
        raise ExecutionError(f"dataset must have {MIN_ROWS}..{MAX_ROWS} rows, has {len(df)}")
    if target not in df.columns:
        raise ExecutionError(f"target column {target!r} not found in dataset")
    n_before = len(df)
    df = df.dropna(subset=[target])
    if len(df) < n_before:
        warnings.append(f"dropped {n_before - len(df)} rows with missing target")
    sha = hashlib.sha256(paths.dataset.read_bytes()).hexdigest()
    return df.drop(columns=[target]), df[target], sha


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=30, sparse_output=False))]), cat))
    return ColumnTransformer(parts)


def split_data(X, y, classification: bool):
    try:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y if classification else None)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=SEED,
            stratify=y_tmp if classification else None)
    except ValueError as e:
        raise ExecutionError(f"cannot split data (too few rows in a class?): {e}") from e
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def positive_index(y_enc: np.ndarray, classes, warnings: list[str]) -> int | None:
    if len(classes) != 2:
        warnings.append("multiclass task: precision/recall/f1 use macro averaging")
        return None
    if 1 in list(classes):
        return list(classes).index(1)
    counts = np.bincount(y_enc)
    pos = int(np.argmin(counts))
    warnings.append(f"no label '1'; minority class {classes[pos]!r} used as positive class")
    return pos


# ---------- models and bounded search spaces ----------

def suggest_params(name: str, trial: optuna.Trial) -> dict:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
    if name in ("random_forest", "random_forest_regressor"):
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)}
    if name in ("gradient_boosting", "gradient_boosting_regressor"):
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 5)}
    raise ExecutionError(f"unsupported model {name!r}")


def build_estimator(name: str, params: dict):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=SEED, **params)
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=1, **params)
    if name == "random_forest_regressor":
        return RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=SEED, **params)
    if name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(random_state=SEED, **params)
    raise ExecutionError(f"unsupported model {name!r}")


def make_pipeline(name: str, params: dict, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(X)), ("model", build_estimator(name, params))])


# ---------- metrics ----------

def compute_metric(metric: str, model, X, y, pos: int | None) -> float:
    if metric in ("rmse", "mae", "r2"):
        pred = model.predict(X)
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y, pred)))
        if metric == "mae":
            return float(mean_absolute_error(y, pred))
        return float(r2_score(y, pred))
    if metric == "roc_auc":
        proba = model.predict_proba(X)
        if pos is not None:
            # la classe positive n'est pas forcément l'indice 1 (étiquettes texte : classe minoritaire)
            return float(roc_auc_score(np.asarray(y) == pos, proba[:, pos]))
        return float(roc_auc_score(y, proba, multi_class="ovr"))
    if metric == "pr_auc":
        if pos is None:
            raise ExecutionError("pr_auc supports binary classification only")
        return float(average_precision_score(np.asarray(y) == pos, model.predict_proba(X)[:, pos]))
    pred = model.predict(X)
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    kw = {"pos_label": pos, "average": "binary"} if pos is not None else {"average": "macro"}
    fn = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[metric]
    return float(fn(y, pred, zero_division=0, **kw))


# ---------- optimization ----------

def cross_val_score(name, params, metric, X, y, pos, classification: bool) -> float:
    """Mean score over CV_FOLDS folds (stratified for classification); the test set is never used."""
    folds = (StratifiedKFold if classification else KFold)(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for fit_idx, eval_idx in folds.split(X, y):
        pipe = make_pipeline(name, params, X.iloc[fit_idx]).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        scores.append(compute_metric(metric, pipe, X.iloc[eval_idx], y.iloc[eval_idx], pos))
    return float(np.mean(scores))


def optimize_model(name, metric, X_tr, y_tr, X_val, y_val, pos, n_trials, timeout_s,
                   classification: bool = True):
    X_dev, y_dev = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        with mlflow.start_run(run_name=f"{name}-trial-{trial.number}", nested=True):
            mlflow.log_params({"model": name, **params})
            val = cross_val_score(name, params, metric, X_dev, y_dev, pos, classification)
            if not np.isfinite(val):
                raise ExecutionError("non-finite score")
            mlflow.log_metric(f"cv_{metric}", val)
            return val

    study = optuna.create_study(
        direction="minimize" if metric in MINIMIZE else "maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return (study.best_params, study.best_value, len(done)) if done else (None, None, 0)


# ---------- quality gate ----------

def quality_gate(metric, threshold, train, val, test, baseline) -> QualityGate:
    s = _sign(metric)
    gt, gv, gs, gb = s * train, s * val, s * test, s * baseline
    gap = (gt - gv) / max(abs(gt), 1e-9)
    drop = (gv - gs) / max(abs(gv), 1e-9)
    checks = {
        "meets_threshold": True if threshold is None
        else (test <= threshold if metric in MINIMIZE else test >= threshold),
        "beats_baseline": gs > gb,
        "overfit_gap_ok": gap <= MAX_REL_OVERFIT_GAP,
        "test_consistent_with_validation": drop <= MAX_REL_TEST_DROP,
    }
    return QualityGate(passed=all(checks.values()), checks=checks,
                       details={"threshold": threshold, "relative_overfit_gap": round(gap, 4),
                                "relative_test_drop": round(drop, 4)})


# ---------- main entry ----------

# Types MLflow's model-loading safety check must accept for our pipelines.
# numpy.dtype is a harmless NumPy type. Add types here only after reviewing them.
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def _log_and_register(pipe, registered_name: str | None, label_classes: list | None = None):
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {}
    if label_classes is not None and "metadata" in params:
        # Le modèle prédit l'indice de la classe (LabelEncoder) : la correspondance voyage AVEC le modèle
        # (fichier MLmodel), pour que l'API rende la bonne étiquette même si un réentraînement change les classes.
        kwargs["metadata"] = {"label_classes": label_classes}
    if registered_name:
        kwargs["registered_model_name"] = registered_name
    if "skops_trusted_types" in params:
        kwargs["skops_trusted_types"] = TRUSTED_TYPES
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    mlflow.sklearn.log_model(pipe, **kwargs)
    if label_classes is not None:
        mlflow.log_dict({"label_classes": label_classes}, "label_classes.json")
    if not registered_name:
        return None, None
    versions = MlflowClient().search_model_versions(f"name='{registered_name}'")
    version = str(max(int(v.version) for v in versions))
    return f"models:/{registered_name}/{version}", version


def _plain(values) -> list:
    """Étiquettes en types Python simples (JSON) : numpy.int64 -> int, etc."""
    return [v.item() if hasattr(v, "item") else v for v in values]


def label_classes_of(model_uri: str) -> list | None:
    """Les classes enregistrées avec un modèle (metadata du MLmodel), ou None (ancien modèle, régression)."""
    try:
        meta = mlflow.models.get_model_info(model_uri).metadata or {}
    except Exception:  # noqa: BLE001 - modèle introuvable ou MLmodel sans metadata
        return None
    classes = meta.get("label_classes")
    return list(classes) if classes is not None else None


def score_champion(champion_uri: str, metric: str, X_test, y_test, classes, pos, warnings: list[str]):
    """Score du champion sur LE MÊME jeu de test que le candidat (sinon on compare deux copies différentes).

    Classification : y_test est encodé avec les classes du candidat ; on repasse par les étiquettes
    d'origine pour l'encoder avec les classes du champion. Renvoie None si le champion ne peut pas prédire
    ces données (colonnes ou classes différentes) : l'appelant garde alors l'ancienne comparaison.
    """
    try:
        champ = mlflow.sklearn.load_model(champion_uri)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"champion non rechargé ({type(e).__name__}) : comparaison sur son ancien score")
        return None
    y_eval, pos_eval = y_test, pos
    if classes is not None:
        champ_classes = label_classes_of(champion_uri) or list(classes)   # ancien modèle : mêmes classes triées
        labels = [classes[int(i)] for i in y_test]
        if not set(labels) <= set(champ_classes):
            warnings.append("le champion ne connaît pas toutes les classes du nouveau jeu de test : "
                            "comparaison sur son ancien score")
            return None
        index = {c: i for i, c in enumerate(champ_classes)}
        y_eval = pd.Series([index[v] for v in labels], index=y_test.index)
        pos_eval = index[classes[pos]] if pos is not None else None
    try:
        return compute_metric(metric, champ, X_test, y_eval, pos_eval)
    except Exception as e:  # noqa: BLE001 - colonnes différentes, etc.
        warnings.append(f"champion non évaluable sur le nouveau jeu de test ({type(e).__name__}) : "
                        "comparaison sur son ancien score")
        return None


def run_experiment(report: AnalysisReport, plan: ExperimentPlan, paths: ResolvedPaths,
                   tracking_uri: str = "sqlite:///mlflow.db",
                   experiment_name: str = "mlflow_agent",
                   registered_model_name: str | None = None,
                   warnings_in: list[str] | None = None,
                   champion_uri: str | None = None) -> ExperimentResult:
    metric = plan.primary_metric
    warnings = list(warnings_in or [])
    result = ExperimentResult(status="failed", primary_metric=metric, plan=plan, warnings=warnings)
    try:
        _run(report, plan, paths, tracking_uri, experiment_name,
             registered_model_name or f"{experiment_name}-model", warnings, result, champion_uri)
    except Exception as e:  # report instead of crash; message truncated
        result.status = "failed"
        result.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
    # pydantic a copié la liste à la création du résultat : sans ceci, les avertissements ajoutés
    # pendant l'exécution (budget de temps, champion, classes...) étaient perdus
    result.warnings = warnings
    return result


def _run(report, plan, paths, tracking_uri, experiment_name, reg_name, warnings, result, champion_uri=None):
    metric = plan.primary_metric
    classification = report.task.type == "classification"
    deadline = time.monotonic() + report.constraints.max_runtime_minutes * 60

    X, y, sha = load_xy(paths, report.task.target_column, warnings)
    pos, classes = None, None
    if classification:
        le = LabelEncoder().fit(y)
        y = pd.Series(le.transform(y), index=y.index)
        classes = _plain(le.classes_)
        pos = positive_index(y.to_numpy(), le.classes_, warnings)
    X_tr, X_val, X_test, y_tr, y_val, y_test = split_data(X, y, classification)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    per_model = max(1, plan.max_trials // len(plan.models))

    with mlflow.start_run(run_name="agent-run") as parent:
        result.run_id = parent.info.run_id
        result.experiment_id = parent.info.experiment_id
        mlflow.log_params({"models": ",".join(plan.models), "max_trials": plan.max_trials,
                           "seed": SEED, "data_sha256": sha[:16], "metric": metric,
                           "selection": f"{CV_FOLDS}-fold cross-validation"})

        best = None  # (name, params, val_score)
        for name in plan.models:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                warnings.append(f"time budget exhausted before trying {name}")
                break
            params, val, done = optimize_model(name, metric, X_tr, y_tr, X_val, y_val,
                                               pos, per_model, remaining, classification)
            result.trials_completed[name] = done
            if params is None:
                warnings.append(f"no successful trial for {name}")
                continue
            if best is None or _sign(metric) * val > _sign(metric) * best[2]:
                best = (name, params, val)
        if best is None:
            raise ExecutionError("no model completed any trial")

        name, params, _ = best
        # generalization check: fit on train only
        pipe_tr = make_pipeline(name, params, X_tr).fit(X_tr, y_tr)
        train_s = compute_metric(metric, pipe_tr, X_tr, y_tr, pos)
        val_s = compute_metric(metric, pipe_tr, X_val, y_val, pos)
        # final model on train+validation; test set is scored exactly once
        X_full, y_full = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
        final = make_pipeline(name, params, X_full).fit(X_full, y_full)
        test_s = compute_metric(metric, final, X_test, y_test, pos)
        dummy = (DummyClassifier(strategy="prior") if classification else DummyRegressor())
        dummy.fit(X_full, y_full)
        base_s = compute_metric(metric, dummy, X_test, y_test, pos)

        champ_s = (score_champion(champion_uri, metric, X_test, y_test, classes, pos, warnings)
                   if champion_uri else None)

        gate = quality_gate(metric, report.task.min_primary_metric, train_s, val_s, test_s, base_s)
        if report.task.min_primary_metric is None:
            warnings.append("no min_primary_metric given: threshold check skipped")

        mlflow.log_metrics({f"train_{metric}": train_s, f"val_{metric}": val_s,
                            f"test_{metric}": test_s, f"baseline_test_{metric}": base_s})
        if champ_s is not None:
            mlflow.log_metric(f"champion_test_{metric}", champ_s)
        uri, version = _log_and_register(final, reg_name if gate.passed else None, classes)

        result.best_model_name, result.best_params = name, params
        result.train_score, result.validation_score = train_s, val_s
        result.test_score, result.baseline_test_score = test_s, base_s
        result.champion_test_score, result.label_classes = champ_s, classes
        result.quality_gate, result.model_uri, result.model_version = gate, uri, version
        result.status = "success" if gate.passed else "no_eligible_model"
        result.reproducibility = {
            "seed": SEED, "data_sha256": sha,
            "split_rows": {"train": len(X_tr), "validation": len(X_val), "test": len(X_test)},
            "test_set_scored_once": True,
            "model_selection": f"{CV_FOLDS}-fold cross-validation on train+validation",
            "versions": {"sklearn": sklearn.__version__, "mlflow": mlflow.__version__,
                         "optuna": optuna.__version__, "pandas": pd.__version__,
                         "numpy": np.__version__},
        }

        # Record the final decision on the parent run, so it is visible in the MLflow UI
        mlflow.log_params({"best_model": name,
                           **{f"best_{k}": v for k, v in params.items()}})
        mlflow.log_metric("gate_passed", float(gate.passed))
        mlflow.set_tags({"agent.status": result.status,
                         "agent.model_uri": str(uri),
                         "agent.plan_rationale": plan.rationale[:250]})
        mlflow.log_dict(result.model_dump(mode="json"), "result.json")
