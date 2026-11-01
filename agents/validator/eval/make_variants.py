"""Génère les projets de test cassés à partir du projet conforme + le fichier des verdicts attendus.
Chaque variante casse UNE seule chose, pour que chaque rejet soit attribuable à une règle précise."""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
BASE = HERE / "projects" / "credit_risk"
OUT = HERE / "projects" / "variants"


def edit(path: Path, fn):
    path.write_text(fn(path.read_text()))


VARIANTS = {
    "01_missing_predict": (lambda d: (d / "predict.py").unlink(), "REJECTED", ["C1.MISSING_FILE"]),
    "02_unpinned_deps": (lambda d: edit(d / "requirements.txt", lambda t: t.replace("scikit-learn==1.8.0", "scikit-learn>=1.0")), "REJECTED", ["C6.UNPINNED"]),
    "03_interactive_main": (lambda d: edit(d / "main.py", lambda t: t.replace("def main():\n", "def main():\n    input('Appuyez sur Entrée')\n")), "REJECTED", ["C5.INTERACTIVE_INPUT"]),
    "04_hardcoded_path": (lambda d: edit(d / "train.py", lambda t: t.replace("RANDOM_STATE = 42", 'RANDOM_STATE = 42\nBACKUP = "/home/user/credit.csv"')), "REJECTED", ["C10.ABSOLUTE_PATH"]),
    "05_dangerous_call": (lambda d: edit(d / "train.py", lambda t: t.replace("import json", "import json\nimport os").replace("    df = pd.read_csv(data_path)", '    os.system("echo hi")\n    df = pd.read_csv(data_path)')), "REJECTED", ["C11.DANGEROUS_CALL"]),
    "06_secret_in_code": (lambda d: edit(d / "train.py", lambda t: t.replace("RANDOM_STATE = 42", 'RANDOM_STATE = 42\nAWS_KEY = "AKIAIOSFODNN7EXAMPLE"')), "REJECTED", ["C11.SECRET"]),
    "07_bad_target": (lambda d: edit(d / "mlops.yaml", lambda t: t.replace("target: target", "target: label")), "REJECTED", ["C7.TARGET_NOT_FOUND"]),
    "08_no_random_state": (lambda d: edit(d / "train.py", lambda t: re.sub(r",\s*random_state=RANDOM_STATE", "", t)), "ACCEPTED_WITH_WARNINGS", ["C9.MISSING_SEED", "C9.NON_DETERMINISTIC"]),
    "09_no_manifest": (lambda d: (d / "mlops.yaml").unlink(), "NEEDS_CONFIRMATION", ["C8.MISSING"]),
    "10_wrong_framework": (lambda d: (edit(d / "requirements.txt", lambda t: t + "xgboost==2.1.4\n"), edit(d / "train.py", lambda t: "import xgboost\n" + t)), "REJECTED", ["C2.UNSUPPORTED_FRAMEWORK"]),
    "11_writes_outside_output": (lambda d: edit(d / "train.py", lambda t: t.replace("    df = pd.read_csv(data_path)", '    open("debug.log", "w").write("x")\n    df = pd.read_csv(data_path)')), "REJECTED", ["C10.WRITES_OUTSIDE_OUTPUT_DIR"]),
    "12_network_import": (lambda d: edit(d / "train.py", lambda t: "import urllib.request\n" + t), "REJECTED", ["C11.NETWORK_IMPORT"]),
    "13_train_crashes": (lambda d: edit(d / "train.py", lambda t: t.replace("    df = pd.read_csv(data_path)", '    raise RuntimeError("boom")\n    df = pd.read_csv(data_path)')), "REJECTED", ["C3.TRAIN_FAILED", "C5.MAIN_FAILED"]),
    "14_wrong_signature": (lambda d: edit(d / "train.py", lambda t: t.replace("def train(data_path, output_dir):", "def train(data_path):").replace("out = Path(output_dir)", 'out = Path("output")')), "REJECTED", ["C3.BAD_SIGNATURE"]),
}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    expected = {
        "credit_risk": {"path": "projects/credit_risk", "verdict": "ACCEPTED", "codes": []},
        "credit_risk_raw": {"path": "projects/credit_risk_raw", "verdict": "REJECTED",
                            "codes": ["C1.MISSING_FILE", "C6.UNPINNED", "C2.UNSUPPORTED_FRAMEWORK", "C8.MISSING", "C11.COMMITTED_ARTIFACT"]},
    }
    for name, (mutate, verdict, codes) in VARIANTS.items():
        dest = OUT / name
        shutil.copytree(BASE, dest, ignore=shutil.ignore_patterns("__pycache__", "output"))
        mutate(dest)
        expected[name] = {"path": f"projects/variants/{name}", "verdict": verdict, "codes": codes}
    (HERE / "expected_verdicts.json").write_text(json.dumps(expected, indent=2))
    print(f"{len(VARIANTS)} variantes générées dans {OUT}")


if __name__ == "__main__":
    main()
