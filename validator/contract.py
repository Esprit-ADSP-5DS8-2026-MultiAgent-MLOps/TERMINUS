"""Le contrat d'entrée (C1..C12) sous forme de configuration explicite et versionnée.

Changer une règle du contrat = modifier ce fichier (ex. autoriser XGBoost), pas la logique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CONTRACT_VERSION = "1.0"
MODEL_ARTIFACT = "model.joblib"  # convention : train() écrit ce fichier dans output_dir

RULES = {
    "C1": "Structure minimale du projet",
    "C2": "Périmètre technique (Python, scikit-learn, données tabulaires)",
    "C3": "train(data_path, output_dir) -> dict de métriques, écrit model.joblib",
    "C4": "load_model(model_path) et predict(model, data)",
    "C5": "main.py non interactif et exécutable sans argument",
    "C6": "Dépendances épinglées (==)",
    "C7": "Données CSV exploitables",
    "C8": "Manifest mlops.yaml valide",
    "C9": "Reproductibilité (graines fixées)",
    "C10": "Chemins relatifs ; écriture limitée à OUTPUT_DIR",
    "C11": "Sécurité : pas de code dangereux, réseau ni secrets",
    "C12": "Limites de ressources",
}

CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1", "roc_auc"})
REGRESSION_METRICS = frozenset({"rmse", "mae", "r2"})

# nom d'import -> nom de distribution pip (quand ils diffèrent)
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml",
    "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "skimage": "scikit-image", "attr": "attrs",
}

NETWORK_MODULES = frozenset({
    "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "ftplib", "smtplib",
    "telnetlib", "paramiko", "websocket", "websockets", "xmlrpc",
})

ARTIFACT_EXTENSIONS = frozenset({".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".onnx", ".pb", ".ckpt"})

# classes/fonctions scikit-learn dont le résultat dépend d'une graine
STOCHASTIC_CALLS = frozenset({
    "train_test_split", "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
    "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor", "DecisionTreeClassifier",
    "DecisionTreeRegressor", "BaggingClassifier", "BaggingRegressor", "MLPClassifier", "MLPRegressor",
    "SGDClassifier", "SGDRegressor", "KMeans", "ShuffleSplit", "StratifiedShuffleSplit",
})
SHUFFLE_SPLITTERS = frozenset({"KFold", "StratifiedKFold"})  # stochastiques seulement si shuffle=True


def norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class ContractConfig:
    required_files: tuple = ("main.py", "train.py", "predict.py", "requirements.txt")
    manifest_name: str = "mlops.yaml"
    default_data_file: str = "data.csv"
    unsupported_frameworks: frozenset = frozenset({
        "torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost", "jax", "jaxlib",
        "transformers", "mxnet", "paddlepaddle",
    })
    allow_frameworks: frozenset = field(default_factory=frozenset)  # ex. {"xgboost"}
    min_rows: int = 50
    max_data_mb: int = 50
    max_project_mb: int = 100
    max_files: int = 500
    missing_rate_warn: float = 0.40
    duplicate_rate_warn: float = 0.20
    dynamic_timeout_s: int = 300
    memory_mb: int = 4096

    @property
    def blocked_frameworks(self) -> frozenset:
        return frozenset(norm_dist(x) for x in self.unsupported_frameworks) - frozenset(
            norm_dist(x) for x in self.allow_frameworks
        )
