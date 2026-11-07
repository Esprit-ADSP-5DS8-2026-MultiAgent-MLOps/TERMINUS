# MLflow Agent

An LLM-assisted experiment runner for validated tabular machine-learning projects. It converts a strict JSON analysis report into a bounded Optuna experiment plan, trains candidate scikit-learn models, logs trials and final results to MLflow, applies a deterministic quality gate, and registers the winning model only when it qualifies.

> **The LLM proposes; the code decides.**

## What it does

1. Validates the input report and execution paths.
2. Asks an Ollama or Gemini model for an experiment plan.
3. Parses and validates the plan against the report's task, metric, model allow-list, and experiment budget.
4. Uses Optuna to tune up to three candidate models.
5. Splits data reproducibly into train, validation, and test sets.
6. Logs every trial and the final decision to MLflow.
7. Applies a deterministic quality gate.
8. Writes an `ExperimentResult` JSON document and optionally registers the model in the MLflow Model Registry.

Existing Python training code and scripts are not executed. The agent owns its preprocessing, estimators, splits, metrics, and training loop.

## Supported tasks

| Task | Models | Metrics |
|---|---|---|
| Classification | Logistic regression, random forest, gradient boosting | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc` |
| Regression | Ridge, random forest regressor, gradient boosting regressor | `rmse`, `mae`, `r2` |

The positive binary-class rule is:

- Label `1`, when present.
- Otherwise, the minority class.

The result and MLflow run record an alternative positive-class warning when label `1` is not present.

## Architecture

```
JSON analysis report
        │
        ▼
Strict schema validation ──► status and path checks
        │
        ▼
LLM plan request (optional)
        │
        ▼
Strict JSON + plan validation
        │                         one invalid response or retry failure
        └──────────────────────────────► deterministic fallback plan
        │
        ▼
Data loading and 60/20/20 split
        │
        ▼
Optuna tuning on the validation set
        │
        ▼
MLflow experiment and trial tracking
        │
        ▼
Final model training + test scoring once
        │
        ▼
Deterministic quality gate
        │
        ├── pass ──► register model + success
        └── fail ──► no registration + no_eligible_model
```

| Component | Responsibility |
|---|---|
| `schemas.py` | Loads and strictly validates reports and resolves safe workspace paths |
| `prompts.py` | Builds the planner's system and user prompts |
| `llm_client.py` | Local Ollama and cloud Gemini clients |
| `config.py` | Resolves the selected LLM provider/model |
| `planner.py` | Requests plans, retries once, and creates the safe fallback |
| `plans.py` | Parses and validates plan JSON |
| `executor.py` | Builds data splits, tuners, models, metrics, MLflow runs, and quality gates |
| `results.py` | Defines the versioned JSON result contract |
| `benchmark.py` / `compare.py` | Measures LLM plan-generation quality |

## Installation

The package supports Python 3.10 and newer; development and validation used Python 3.12.

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,agent]"
```

The `agent` extra installs scikit-learn, Optuna, pandas, NumPy, and MLflow.

## Input report

The default example is [`examples/credit_risk_report.json`](../examples/credit_risk_report.json).

The report must contain:

- `schema_version` — currently `0.1`
- `project.project_path` and `project.dataset_path`
- `project.training_script`, if applicable
- `task.type`, `task.target_column`, `task.primary_metric`, and optional `task.min_primary_metric`
- Dataset size, class distribution, missing values, and leakage warnings
- Existing implementation information
- `constraints.max_experiment_runs` and `constraints.max_runtime_minutes`
- `status: "ready_for_experimentation"`

Unknown JSON fields, inconsistent metrics, unsupported values, excessive report sizes, and unavailable or unsafe paths are rejected.

### Report example

```json
{
  "schema_version": "0.1",
  "project": {
    "project_path": ".",
    "dataset_path": "data.csv",
    "training_script": null
  },
  "task": {
    "type": "classification",
    "target_column": "target",
    "objective": "credit-risk",
    "primary_metric": "roc_auc",
    "min_primary_metric": 0.70
  },
  "dataset_analysis": {
    "rows": 1000,
    "columns": 21,
    "missing_values": {"total": 0, "columns_affected": []},
    "class_distribution": {"1": 700, "0": 300},
    "data_leakage_risks": []
  },
  "existing_implementation": {
    "training_code_available": true,
    "preprocessing_available": true,
    "existing_models": []
  },
  "constraints": {
    "max_experiment_runs": 12,
    "max_runtime_minutes": 10
  },
  "status": "ready_for_experimentation"
}
```

`secondary_metrics`, `warnings`, and missing-value metadata may be included according to the schema.

## Choosing an LLM

Set the model in `mlflow_agent.toml`:

```toml
llm = "qwen2.5:3b"
# llm = "gemini:gemini-3.1-flash-lite"
# llm = "ollama:llama3.2:3b"
```

Resolution order:

1. `--model`
2. `MLFLOW_AGENT_LLM` environment variable
3. `mlflow_agent.toml`
4. Built-in default: `qwen2.5:3b`

The selected model and source are printed at the start of each run.

### Ollama

```bash
ollama pull qwen2.5:3b
ollama serve
python -m mlflow_agent examples/credit_risk_report.json --model qwen2.5:3b
```

The Ollama client only accepts a localhost server.

### Gemini

```bash
export GEMINI_API_KEY="your-key"
python -m mlflow_agent examples/credit_risk_report.json --model gemini:gemini-3.1-flash-lite
```

Multiple Gemini models can be supplied as a comma-separated pool:

```bash
python -m mlflow_agent examples/credit_risk_report.json \
  --model gemini:gemini-3.5-flash-lite,gemini-3.1-flash-lite
```

If one pooled model returns a quota error, it is cooled down and the next model is tried. Gemini 404s are skipped for the rest of the run.

## Running the agent

The commands below assume the repository root and a valid dataset beneath the selected base directory.

### Generate and validate a plan only

```bash
python -m mlflow_agent examples/credit_risk_report.json
```

This validates the report, requests the model's plan, validates it, and prints the result. It does not train models.

### Run training and tuning

```bash
python -m mlflow_agent examples/credit_risk_report.json \
  --base-dir . \
  --execute \
  --out result.json \
  --experiment mlflow_agent
```

For this example, `examples/credit_risk_report.json` resolves `.` + `data.csv`, so the dataset must exist at the repository root as `data.csv` or be supplied through a correctly configured base directory.

### Use a deterministic plan without an LLM

```bash
python -m mlflow_agent examples/credit_risk_report.json \
  --base-dir . \
  --execute \
  --no-llm
```

The fallback plan uses:

- Classification: logistic regression and random forest
- Regression: ridge and random forest regressor
- At most 20 runs, and never more than the report budget

### CLI options

| Option | Description |
|---|---|
| `report` | Path to the input JSON report |
| `--model` | Override the configured Ollama or Gemini model |
| `--no-llm` | Skip planning calls and use the deterministic fallback |
| `--execute` | Load the dataset and run the experiments |
| `--base-dir` | Root used to resolve report paths |
| `--out` | Output path for the result JSON |
| `--experiment` | MLflow experiment name |

## Training and data handling

- Random seed: `42`
- Dataset limit: 50 MB and 50–200,000 rows
- Split: 60% train, 20% validation, 20% test
- Numerical features: median imputation and standard scaling
- Categorical features: most-frequent imputation and one-hot encoding with an infrequent-category limit
- Classification and regression models are trained from deterministic scikit-learn pipelines
- Trial runs are nested under a parent MLflow run
- The test set is scored exactly once after final training
- The final model is fit on train plus validation before test scoring
- Each trial uses only the validation split for model selection

The report's runtime budget caps the experiment loop. Each model receives a bounded share of the requested trials; execution stops if the deadline is reached.

## MLflow tracking

By default, the agent writes to `sqlite:///mlflow.db` and the experiment named `mlflow_agent`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to inspect:

- Trial parameters and validation metrics
- Train, validation, test, and dummy-baseline scores
- The quality-gate result
- The saved `result.json`
- The registered model, when the gate passes

The registration name is `<experiment-name>-model` by default, for example `mlflow_agent-model`.

## Quality gate

A candidate is eligible only when every applicable check passes:

| Check | Rule |
|---|---|
| Minimum metric | Meets `task.min_primary_metric`; skipped if not supplied |
| Beats baseline | Outperforms a dummy prior/constant baseline |
| Overfit control | At most half of the model's gain over the naive model (measured on train) is lost on validation: `(train − val) / (train − naive_val)`, same scale for scores and errors (rmse, mae) |
| Test consistency | Relative validation–test drop is no greater than 25% |

The quality gate is plain deterministic Python. The LLM does not participate in the decision.

Model selection respects the overfit control: models are ranked by cross-validation score, and the best one
that passes the overfit control is kept (a model that loses more than half of its gain on validation gives way
to the next one, with a warning). This uses train and validation only: the test set is still scored once, by
the chosen model.

When the gate passes, the final pipeline is registered in MLflow. When it fails, no model is registered and the result status is `no_eligible_model`.

## Result file

The agent writes a versioned JSON result similar to:

```json
{
  "schema_version": "0.1",
  "status": "success",
  "primary_metric": "roc_auc",
  "best_model_name": "gradient_boosting",
  "validation_score": 0.8117,
  "test_score": 0.7706,
  "baseline_test_score": 0.5,
  "model_uri": "models:/mlflow_agent-model/1",
  "quality_gate": {
    "passed": true,
    "checks": {
      "meets_threshold": true,
      "beats_baseline": true,
      "overfit_gap_ok": true,
      "test_consistent_with_validation": true
    }
  },
  "trials_completed": {
    "gradient_boosting": 4,
    "logistic_regression": 4,
    "random_forest": 4
  }
}
```

Possible top-level statuses are:

- `success` — quality gate passed and the model was registered
- `no_eligible_model` — experiments ran, but the quality gate failed
- `failed` — execution could not complete, with details in `errors`

The complete result also includes the plan, parameters, train/validation/test scores, MLflow IDs, quality-gate details, warnings, errors, dataset SHA-256, split sizes, random seed, and library versions.

## Security boundaries

- Reports are limited to 1 MB and strict Pydantic models reject unknown fields.
- Plans reject unknown models, duplicate models, wrong metrics, wrong search methods, and budgets above hard limits.
- Invalid LLM plans get one retry; if still invalid, the deterministic fallback plan is used.
- LLM project text is fenced inside an untrusted-data prompt section and cannot change planner invariants.
- Project datasets are resolved beneath a chosen base directory; absolute paths, parent traversal, and escaping symlinks are rejected.
- Execution reads CSV data only; project training scripts are never imported or run.
- Gemini API keys come only from environment variables and are redacted from error output.
- Gemini keys are sent in the API header, never in the URL.
- Ollama hosts are restricted to localhost.
- Dataset size, row count, runtime, trial count, report size, and plan size have hard limits.

## Benchmarking the planner

The benchmark suite checks raw LLM output without retries and without a fallback. It uses fixed normal and prompt-injection cases and verifies valid JSON, schema compliance, task-compatible models, uniqueness, the required metric, and the experiment budget.

```bash
python -m mlflow_agent.benchmark --model qwen2.5:3b --prompt v1-structured
python -m mlflow_agent.compare
python -m mlflow_agent.compare --md
```

Available prompt variants are `v0-basic`, `v1-structured`, `v2-fewshot`, and `v3-guard`.

Prompt performance depends on the provider and model, so benchmarks should be rerun for deployment-specific models rather than assuming a prompt version is universally best.

## Tests

Run the focused planner/executor tests or the complete suite:

```bash
python -m pytest -q tests/test_mlflow_agent_input.py
python -m pytest -q tests/test_plans.py tests/test_planner.py tests/test_executor.py
python -m pytest -q
```

The test suite covers strict report validation, path containment, plan parsing and fallback, prompt construction, LLM clients, execution, model registration, benchmark output, and result contracts.

## Example workflow

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Configure an Ollama model
ollama pull qwen2.5:3b
ollama serve

# 3. Optionally inspect the plan
python -m mlflow_agent examples/credit_risk_report.json \
  --model qwen2.5:3b

# 4. Run the bounded experiment
python -m mlflow_agent examples/credit_risk_report.json \
  --base-dir . \
  --execute \
  --model qwen2.5:3b \
  --out result.json

# 5. Inspect MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
