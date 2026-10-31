import os

import pandas as pd

from predict import load_model, predict
from train import train


def main():
    input('Appuyez sur Entrée')
    out = os.environ.get("OUTPUT_DIR", "output")
    print("metrics:", train("data.csv", out))
    model = load_model(os.path.join(out, "model.joblib"))
    sample = pd.read_csv("data.csv").drop(columns=["target"]).head(3)
    print("predictions:", predict(model, sample))


if __name__ == "__main__":
    main()
