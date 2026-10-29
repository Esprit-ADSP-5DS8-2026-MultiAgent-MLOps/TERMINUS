import pandas as pd  # nettoyé

DATA = "data.csv"


def main():
    print(pd.read_csv(DATA).head())


if __name__ == "__main__":
    main()
