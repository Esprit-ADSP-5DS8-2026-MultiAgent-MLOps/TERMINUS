"""Projet minimal généré dans un dossier temporaire : les tests ne dépendent d'aucun projet nommé du dépôt."""
from pathlib import Path

REQ_LINE = "scikit-learn==1.5.2"
REQUIREMENTS = f"{REQ_LINE}\npandas==2.2.2\n"
TRAIN = '''import pandas as pd
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def train(data_path="data.csv", target="target"):
    df = pd.read_csv(data_path)
    model = LogisticRegression(random_state=RANDOM_STATE).fit(df.drop(columns=[target]), df[target])
    return model, {"accuracy": float(model.score(df.drop(columns=[target]), df[target]))}
'''


def make_mini_project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (root / "train.py").write_text(TRAIN, encoding="utf-8")
    (root / "main.py").write_text("from train import train\n\nif __name__ == '__main__':\n    print(train()[1])\n", encoding="utf-8")
    (root / "data.csv").write_text("f1,f2,target\n1,2,0\n2,1,1\n3,5,0\n4,3,1\n5,8,0\n6,2,1\n", encoding="utf-8")
    (root / "mlops.yaml").write_text("data:\n  path: data.csv\n  target: target\n", encoding="utf-8")
    return root