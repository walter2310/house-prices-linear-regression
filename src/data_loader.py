import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/raw/housing.csv'

def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def train_test_split_data(df, rstate=2, shuffle=True, stratify=None):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"].copy()

    strat = df[stratify] if stratify else None
    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.3, random_state=rstate, shuffle=shuffle, stratify=strat)

    return x_train_set, x_test_set, y_train_set, y_test_set


def check_data(df):
    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    # Cargar datos
    df = load_data()
    check_data(df)

    # Dividir datos
    train_test_split_data(df, stratify='ocean_proximity')