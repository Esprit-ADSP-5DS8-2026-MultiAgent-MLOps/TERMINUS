import pandas as pd

DATA = "/home/user/project/data.csv"


def main():
    print(pd.read_csv(DATA).head())


if __name__ == "__main__":
    main()
