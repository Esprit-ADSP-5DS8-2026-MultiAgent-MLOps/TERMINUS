"""Séparation variables / cible."""


def split_xy(df, target):
    return df.drop(columns=[target]), df[target]
