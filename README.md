# Contract Validator — Agentic MLOps

The **Contract Validator** is the entry point of the platform. It checks that a Machine Learning project complies with a **contract** (structure, dependencies, safe code, usable data…) before the agents take it over.

- **No LLM**: the same project always gets the same verdict.
- **Safe**: the project's code is only executed once it has passed all the non-executing analyses, and then only in an isolated environment.
- **It produces a JSON report** (`compatibility_report.json`) that serves as input for the next agent.

```
ML project ──► Contract Validator ──► JSON report ──► Agent 1 ──► …
               (accepted / to confirm / rejected)     (if the handoff is allowed)
```

---

## 1. Prerequisites

| | |
|---|---|
| System | **Ubuntu 24.04 LTS** recommended · compatible with **WSL2** (Windows 10/11) · x86_64 / amd64 architecture |
| Python | **≥ 3.12** (validated version: 3.12.3). Python 3.8 is **not** supported |
| Tools | `git`, `python3-venv`, `python3-pip`, `build-essential` |
| Network | Internet for installation, and on the **first** validation run (download of the tested project's dependencies) |

Windows without WSL is not supported (the validator uses Linux features).

Reference versions used during development: Ubuntu 24.04.5 LTS · Python 3.12.3 · pip 24.0+ · Git 2.43.0+

---

## 2. Installation

**1. Install the system tools** (once only):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git build-essential
```

**2. Get the code**:

```bash
mkdir -p ~/projects && cd ~/projects
git clone <REPOSITORY_URL> agentic-mlops
cd agentic-mlops
git checkout feature/contract-validator      # if the branch has not yet been merged into main
```

**3. Create the validator's virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> To be repeated in every new terminal: `source .venv/bin/activate` (the prompt shows `(.venv)`).

**4. Install the project and its development dependencies**:

```bash
python -m pip install -e ".[dev]"
```

**5. Verify**:

```bash
python --version        # must show 3.12 or higher
python -m pip --version
git --version
python -m pytest -q     # 41 tests, all must pass
```

The tests take **1 to 3 minutes the first time**: the validator downloads scikit-learn, pandas, etc. for the test project, then keeps them in cache (`~/.cache/contract-validator`). Subsequent runs: about 30 seconds.

> **WSL2 — TLS errors or failing downloads?** On some networks, you need to lower the MTU to 1400:
> `sudo ip link set dev eth0 mtu 1400` (valid until WSL restarts).

---

## 3. Usage in 3 commands

Always from the repository root, with the environment activated.

```bash
# 1) Validate a project
python -m validator eval/projects/credit_risk

# 2) Validate and produce the report for the next agent (+ frozen copy of the project)
python -m validator eval/projects/credit_risk \
    --json compatibility_report.json --snapshot-dir ~/validated

# 3) Check that a report can be used (this is the next agent's check)
python -m validator.handoff verify compatibility_report.json
```

The first run on a new project takes 20 to 60 seconds (creation of its isolated environment); subsequent runs take about 5 seconds.

To validate **your own** project: `python -m validator /path/to/my_project` (anywhere on disk).

---

## 4. Reading the result

### The 4 verdicts

| Verdict | Meaning | Exit code |
|---|---|---|
| `ACCEPTED` | Compliant | `0` |
| `ACCEPTED_WITH_WARNINGS` | Compliant, with points to watch | `0` |
| `NEEDS_CONFIRMATION` | A human must confirm something (e.g. `mlops.yaml` missing) | `10` |
| `REJECTED` | Non-compliant: at least one blocker | `20` |

### The 4 severity levels of a finding

`✖ BLOCKER` rejects · `? CONFIRM` requests a confirmation · `! WARNING` to be watched · `i INFO` simple information.

### Example output

```
Verdict: ACCEPTED   (0 blocker(s), 0 confirmation(s), 0 warning(s))   [4.1s]
Levels: structure ✔ | manifest ✔ | static ✔ | data ✔ | dynamic ✔
Handoff: READY (can proceed: yes)
Observed metrics: accuracy=0.7150, ..., roc_auc=0.7773
```

Each finding gives a **stable code** (e.g. `C6.UNPINNED`), the file and line, a **piece of evidence** and a **suggested fix** (`hint`). Example:

```
[✖ BLOCKER] C6.UNPINNED requirements.txt:3 — Unpinned dependency: scikit-learn>=1.3
        hint   : Pin the exact version, e.g. `scikit-learn==<tested version>`.
```

### The 5 control levels

| # | Level | Executes the project's code? |
|---|---|---|
| 1 | Structure and dependencies | No |
| 2 | `mlops.yaml` manifest | No |
| 3 | Code analysis (AST, secrets, Bandit) | No |
| 4 | Data (CSV) | No |
| 5 | Control run in an isolated environment | **Yes, only if levels 1 to 4 have no blockers** |

---

## 5. Preparing an ML project so that it is accepted

### What the project folder must contain

```
my_project/
├── main.py            # runs everything, with no argument or input
├── train.py           # def train(data_path, output_dir) -> dict of metrics
├── predict.py         # def load_model(model_path) and def predict(model, data)
├── requirements.txt   # pinned versions: package==version
├── data.csv           # data (UTF-8, with the target column)
└── mlops.yaml         # task, target, metric
```

### The contract in plain language

| Rule | In plain language |
|---|---|
| C1 | The 4 files `main.py`, `train.py`, `predict.py`, `requirements.txt` are **at the root** |
| C2 | Python + **scikit-learn** + pandas. No PyTorch, TensorFlow, XGBoost… (can be changed, see §7) |
| C3 | `train(data_path, output_dir)` returns a **dict** containing the manifest's metric and writes `model.joblib` to `output_dir` |
| C4 | `load_model(model_path)`; `predict(model, data)` accepts a **dict** *and* a **DataFrame**, and returns a **list** |
| C5 | `main.py` runs with no argument, **without `input()`**, and writes to `$OUTPUT_DIR` |
| C6 | All dependencies are pinned with `==` |
| C7 | CSV in UTF-8, at least 50 rows, target column present, at least 2 classes |
| C8 | `mlops.yaml` present and valid (absent = human confirmation requested) |
| C9 | Fixed seeds (`random_state=...`): two trainings give the same result |
| C10 | No absolute paths; **nothing is written outside `output_dir`** |
| C11 | No `eval`/`exec`/`os.system`, no network, no secrets in the code, no committed `.pkl` |
| C12 | Limits: 500 files, 100 MB, 50 MB of data, 300 s of execution |

### Example `mlops.yaml`

```yaml
project:
  name: credit-risk
  task: classification          # classification | regression
data:
  path: data.csv
  target: target                # name of the column to predict
evaluation:
  primary_metric: roc_auc       # classification: accuracy, precision, recall, f1, roc_auc
  min_threshold: 0.70           # regression: rmse, mae, r2
```

### Minimal example of the expected functions

```python
# train.py
def train(data_path, output_dir):
    ...                                       # trains, writes output_dir/model.joblib
    return {"roc_auc": 0.78, "f1": 0.79}      # must contain the manifest's metric

# predict.py
def load_model(model_path): ...
def predict(model, data): ...                 # data: dict, list of dicts or DataFrame -> list
```

A complete, compliant project is provided: `eval/projects/credit_risk/`.

---

## 6. The report and the handoff

The `compatibility_report.json` report contains the verdict, the detailed findings, and above all a **`handoff`** section: the summary intended for the next agent.

| Element | Role |
|---|---|
| `handoff.can_proceed` | Is the next agent allowed to start? |
| `handoff` (task, target, metric, input columns, dependencies, observed metrics) | What the validator has already established: the agent does not need to redo it |
| `handoff.carry_forward` | Warnings to pass on (imbalanced classes, small dataset…) |
| `project_hash` | Fingerprint of the validated project: a single modification changes it |
| `snapshot_path` | Read-only frozen copy of the validated project (with `--snapshot-dir`) |
| `report_sha256` | Seal: detects any modification of the report |

### The 5 handoff statuses

| Status | Meaning | Can the next agent start? |
|---|---|---|
| `READY` / `READY_WITH_WARNINGS` | Compliant and executed | Yes |
| `AWAITING_CONFIRMATION` | Human confirmation required | No |
| `INCOMPLETE` | No error, but the project was **not executed** (`--no-dynamic`) | No |
| `BLOCKED` | Rejected | No |

> The orchestrator must read **`can_proceed`** (or call `verify_handoff`), **not just the verdict**: with `--no-dynamic`, the verdict can be `ACCEPTED` while the handoff is refused.

### Agent / orchestrator side

```python
from validator.handoff import verify_handoff

check = verify_handoff("compatibility_report.json", project_dir)   # project_dir optional if snapshot_path
if not check.ok:
    raise SystemExit(check.reasons)        # human-readable list of the reasons for the refusal
```

`verify_handoff` checks: the format version, the report's integrity, `can_proceed`, and that the project has not changed since validation. JSON schema of the report: `docs/compatibility_report.schema.json`.

---

## 7. Useful options

| Option | Effect |
|---|---|
| `--json FILE` | Writes the JSON report |
| `--snapshot-dir FOLDER` | Frozen copy of the validated project (created only if the handoff is allowed) |
| `--no-dynamic` | Never executes the project's code (analyses 1 to 4 only) |
| `--allow-framework xgboost` | Extends the contract to a framework (repeatable) |
| `--env venv` (default) / `--env current` | `venv`: isolated environment built from the project's `requirements.txt`. `current`: current Python (not recommended) |
| `--cache-dir FOLDER` | Location of the environment cache (default `~/.cache/contract-validator`, or the `CV_CACHE_DIR` variable) |
| `--timeout N` | Execution limit in seconds (default 300) |
| `--no-determinism` | Skips the second control training |
| `--no-bandit` | Disables Bandit |
| `--confirmed-by "Name"` | Records who confirmed the manifest (simple trace: **never** changes the verdict) |
| `-q` | Quiet mode (useful with `--json` or in CI) |

---

## 8. Repository layout

```
agentic-mlops/
├── README.md
├── pyproject.toml
├── validator/                  # the validator's code
│   ├── __main__.py             # command line
│   ├── engine.py               # chains the levels and computes the verdict
│   ├── contract.py             # the C1–C12 contract (modify to change a rule)
│   ├── models.py               # report and manifest schemas
│   ├── check_structure.py      # level 1: files, dependencies
│   ├── check_manifest.py       # level 2: mlops.yaml
│   ├── check_static.py         # level 3: code analysis without executing it
│   ├── check_data.py           # level 4: CSV
│   ├── check_dynamic.py        # level 5: control run
│   ├── sandbox.py              # isolated environment (venv + Docker)
│   ├── harness.py              # script executed in the sandbox
│   ├── handoff.py              # handoff, fingerprint, seal
│   ├── context.py  fs.py       # shared state, safe file reading
├── tests/                      # 41 tests
├── eval/
│   ├── make_variants.py        # generates the broken projects for the tests
│   └── projects/
│       ├── credit_risk_raw/    # original project, non-compliant (intact copy)
│       ├── credit_risk/        # compliant version (reference)
│       └── variants/           # 14 broken variants (generated)
└── docs/
    └── compatibility_report.schema.json
```

---

## 9. Tests and sample projects

```bash
python -m pytest -q                         # everything
python -m pytest tests/test_handoff.py -q   # the handoff only
python eval/make_variants.py                # regenerates the 14 broken variants
```

The tests validate **16 projects whose verdict is known in advance**: the reference must be accepted, the raw project rejected, and each broken variant must receive exactly the expected verdict (for example `02_unpinned_deps` → `C6.UNPINNED`). They also check the sandbox: infinite loop stopped, network blocked, secret variables not passed through.

**After modifying `models.py`**, regenerate the schema (a test fails otherwise):

```bash
python -m validator.handoff schema docs/compatibility_report.schema.json
```

---

## 10. Common problems

| Symptom | Cause and solution |
|---|---|
| `externally-managed-environment` | The virtual environment is not activated. Run `source .venv/bin/activate`. Do not use `--break-system-packages` |
| `No module named validator` | Environment not activated, or `pip install -e ".[dev]"` not done, or command run outside the repository root |
| `Multiple top-level packages discovered` during installation | `pyproject.toml` must contain `[tool.setuptools]` and `packages = ["validator"]` |
| `ensurepip is not available` / venv creation impossible | `sudo apt install -y python3-venv` |
| `C6.INSTALL_FAILED` | A pinned version does not exist for Python 3.12, or there is no network. Read the `evidence` field |
| Very long first run | Normal: creation and filling of the cache. To start from scratch: `rm -rf ~/.cache/contract-validator` |
| Failing downloads or TLS (WSL2) | `sudo ip link set dev eth0 mtu 1400` |
| `Network not isolated in this sandbox` (INFO) | Ubuntu blocks `unshare`. This is not blocking, but the validator is then less watertight. Test with: `unshare -rn true` |
| Tests failing without Internet | The first run needs the network to build the test project's environment |
| `INCOMPLETE` while the verdict is `ACCEPTED` | You used `--no-dynamic`: the project was not executed, so there is no handoff |

---

## 11. Known limitations

- **The Docker backend is written but has not been tested.** Only the `subprocess` fallback is covered by the tests.
- This fallback is **not** a complete security barrier: network cut off and resources limited, but neither a read-only file system nor a non-root user. Use Docker in production.
- Static analysis cannot see deliberately obfuscated code: this is why the code is only executed in the sandbox.
- Secrets are detected using a list of patterns (no gitleaks).
- The report's **seal** detects accidental modifications; it does not authenticate its author (no signature).
- `--confirmed-by` does not authenticate the person: it is a declaration, recorded as a trace.
- Linux and macOS only (on Windows: WSL2).

---

## 12. Working as a team

```bash
git checkout -b feature/my-change
# ... changes ...
python -m pytest -q                 # must stay green
git add .
git commit -m "feat(validator): short description"
git push -u origin feature/my-change
```

Then open a Pull Request to `main`. To change a contract rule, modify `validator/contract.py` and add a broken variant in `eval/make_variants.py`: it serves as a regression test.