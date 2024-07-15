# %%
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

TRAIN_SET_PATH = '../data/processed/train_set.csv'
MODEL_FILENAME = '../models/linear_regression_model.pkl'

df = pd.read_csv(TRAIN_SET_PATH)

x_train_set = df.drop("median_house_value", axis=1)
y_train_set = df["median_house_value"].copy()

lin_reg = LinearRegression()

lin_reg.fit(x_train_set, y_train_set)

joblib.dump(lin_reg, MODEL_FILENAME)