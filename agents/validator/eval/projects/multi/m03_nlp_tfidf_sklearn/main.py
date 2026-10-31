import os

import pandas as pd

from predict import load_model, predict
from train import train


def main():
    out = os.environ.get("OUTPUT_DIR", "output")
    print("metrics:", train("data.csv", out))
    model = load_model(os.path.join(out, "model.joblib"))
    print("predictions:", predict(model, pd.read_csv("data.csv").drop(columns=["label"]).head(3)))


if __name__ == "__main__":
    main()
