import numpy as np
import pandas as pd
from scipy import stats

# Función para manejar valores faltantes
def handle_missing_values(df):
    return df.dropna()

# Función para transformar columnas específicas con una transformación logarítmica
def apply_log_transform(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = np.log(df[col] + 1) # Utilizamos el +1 para evitar log(0) que es indeterminado
    return df

# Función para transformar valores categóricos en numéricos usando One-Hot Encoding
def apply_one_hot_encoding(df, categorical_column):
    dummies = pd.get_dummies(df[categorical_column]).astype(int)
    df = df.join(dummies).drop([categorical_column], axis=1)
    return df

def remove_outliers_zscore(df, threshold=3):
    z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    df_filtered = df[filtered_entries]

    return df_filtered