"""Défauts injectés dans un projet sain. Chaque défaut casse UNE chose ; le projet d'origine sert de référence (« or »).

Seuls figurent les défauts qui ARRIVENT au LLM du Repair : statut REPAIR_REQUIRED, natures mécanique / configuration (politique).
Les autres ne servent pas à entraîner le modèle, car Python tranche AVANT l'appel au LLM (agent.py : triage) :
  - sécurité (appel dangereux, secret, réseau)      -> REJECTED, aucune réparation tentée
  - graine absente, artefact commité, framework hors périmètre, métriques écrites en dur -> NEEDS_REVIEW (revue humaine, pas de REPAIR)
  - problèmes de nature logique / méthodologique    -> jamais confiés au LLM (listés « needs_human »)
`expected_checks` est vérifié par les tests : le Validator doit bien signaler ces contrôles sur le projet cassé.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

def _edit(path: Path, fn: Callable[[str], str]) -> None:
    text = path.read_text(encoding="utf-8")
    new = fn(text)
    if new == text:
        raise ValueError(f"injection sans effet sur {path.name}")
    path.write_text(new, encoding="utf-8")


def _before_read_csv(code: str) -> Callable[[str], str]:
    return lambda t: t.replace("    df = pd.read_csv(data_path)", f"{code}\n    df = pd.read_csv(data_path)", 1)


def _abs_dead(d, rng, meta):
    name = meta["domain"].replace("-", "_")
    _edit(d / "train.py", lambda t: t.replace(f"RANDOM_STATE = {meta['random_state']}", f'RANDOM_STATE = {meta["random_state"]}\nBACKUP = "/home/user/{name}.csv"', 1))


def _abs_used(d, rng, meta):
    name = meta["domain"].replace("-", "_")
    _edit(d / "main.py", lambda t: t.replace('train("data.csv", out)', f'train("/home/user/{name}.csv", out)').replace('pd.read_csv("data.csv")', f'pd.read_csv("/home/user/{name}.csv")'))


def _unpinned(d, rng, meta):
    pkg = rng.choice(["scikit-learn", "pandas", "numpy", "joblib"])
    how = rng.choice([">=", "~=", "bare"])

    def fn(text: str) -> str:
        def sub(m: re.Match) -> str:
            return pkg if how == "bare" else f"{pkg}{how}{m.group(1)}"
        return re.sub(r"^" + re.escape(pkg) + r"==([\w.]+)", sub, text, flags=re.M)
    _edit(d / "requirements.txt", fn)


def _interactive(d, rng, meta):
    _edit(d / "main.py", lambda t: t.replace("def main():\n", "def main():\n    input('Appuyez sur Entrée')\n", 1))


def _syntax_error(d, rng, meta):
    file, old, new = rng.choice([("predict.py", "def load_model(model_path):", "def load_model(model_path)"),
                                 ("train.py", "def _models():", "def _models()"),
                                 ("preprocessing.py", '("model", model)])', '("model", model)')])
    _edit(d / file, lambda t: t.replace(old, new, 1))


def _bad_data_path(d, rng, meta):
    _edit(d / "mlops.yaml", lambda t: t.replace("path: data.csv", f"path: {rng.choice(['data/train.csv', 'dataset.csv', 'data/' + meta['domain'] + '.csv'])}"))


def _predict_signature(d, rng, meta):
    _edit(d / "predict.py", lambda t: t.replace("def predict(model, data):", "def predict(data):"))


def _no_return(d, rng, meta):
    _edit(d / "train.py", lambda t: t.replace("    return results[best]\n", ""))


def _no_model_saved(d, rng, meta):
    _edit(d / "train.py", lambda t: t.replace('    joblib.dump(fitted[best], out / "model.joblib")\n', ""))


def _typo_import(d, rng, meta):
    _edit(d / "train.py", lambda t: t.replace("from preprocessing import build_pipeline", "from preprocesing import build_pipeline"))


def _missing_import(d, rng, meta):
    _edit(d / "train.py", lambda t: t.replace("import joblib\n", "", 1))


def _writes_outside(d, rng, meta):
    _edit(d / "train.py", _before_read_csv('    open("debug.log", "w").write("x")'))


def _missing_predict(d, rng, meta):
    (d / "predict.py").unlink()


def _wrong_signature(d, rng, meta):
    _edit(d / "train.py", lambda t: t.replace("def train(data_path, output_dir):", "def train(data_path):").replace("out = Path(output_dir)", 'out = Path("output")'))


def _bad_target(d, rng, meta):
    _edit(d / "mlops.yaml", lambda t: t.replace(f"target: {meta['target']}", "target: label"))


def _train_crashes(d, rng, meta):
    _edit(d / "train.py", _before_read_csv('    raise RuntimeError("boom")'))


@dataclass(frozen=True)
class Fault:
    id: str
    kind: str                       # repair (seul type conservé : voir docstring)
    description: str
    apply: Callable[[Path, random.Random, dict], None]
    expected_checks: tuple[str, ...] = field(default_factory=tuple)      # contrôles que le Validator doit signaler (voir tests)
    gold_files: tuple[str, ...] = ()                                     # fichiers touchés par l'injection (référence « or » = leur état sain)


FAULTS: dict[str, Fault] = {f.id: f for f in [
    Fault("abs_path_dead", "repair", "chemin absolu dans une constante inutilisée", _abs_dead, ("IO.PATHS",), ("train.py",)),
    Fault("abs_path_used", "repair", "chemin absolu réellement lu par main.py", _abs_used, ("IO.PATHS",), ("main.py",)),
    Fault("unpinned_dep", "repair", "dépendance non épinglée", _unpinned, ("ENV.DEPENDENCIES_PINNED",), ("requirements.txt",)),
    Fault("interactive_input", "repair", "input() bloquant dans main", _interactive, ("ML.NON_INTERACTIVE",), ("main.py",)),
    Fault("writes_outside_output", "repair", "écriture hors du dossier de sortie", _writes_outside, ("IO.PATHS",), ("train.py",)),
    Fault("missing_predict", "repair", "predict.py manquant", _missing_predict, ("STRUCT.REQUIRED_FILES",), ("predict.py",)),
    Fault("wrong_signature", "repair", "train() sans output_dir", _wrong_signature, ("ML.TRAIN_INTERFACE",), ("train.py",)),
    Fault("syntax_error", "repair", "erreur de syntaxe Python", _syntax_error, ("CODE.SYNTAX",), ("predict.py", "preprocessing.py", "train.py")),
    Fault("bad_data_path", "repair", "chemin des données du manifeste inexistant", _bad_data_path, ("DATA.USABLE",), ("mlops.yaml",)),
    Fault("predict_signature", "repair", "predict(data) sans model", _predict_signature, ("ML.PREDICT_INTERFACE",), ("predict.py",)),
    Fault("no_return", "repair", "train() ne renvoie pas les métriques", _no_return, ("EXEC.RUN",), ("train.py",)),
    Fault("no_model_saved", "repair", "train() n'écrit pas model.joblib", _no_model_saved, ("EXEC.RUN",), ("train.py",)),
    Fault("typo_import", "repair", "import d'un module inexistant (faute de frappe)", _typo_import, ("EXEC.RUN",), ("train.py",)),
    Fault("missing_import", "repair", "import joblib oublié", _missing_import, ("EXEC.RUN",), ("train.py",)),
    Fault("train_crashes", "repair", "entraînement qui lève une exception", _train_crashes, ("EXEC.RUN",), ("train.py",)),
    Fault("bad_target", "repair", "cible du manifeste inexistante", _bad_target, ("DATA.USABLE",), ("mlops.yaml",)),
]}


def inject(fault_id: str, dest: Path, seed: int, meta: dict) -> Fault:
    """Applique le défaut dans `dest` (copie du projet sain) ; déterministe pour (fault_id, seed)."""
    f = FAULTS[fault_id]
    f.apply(dest, random.Random(f"{fault_id}:{seed}"), meta)
    return f
