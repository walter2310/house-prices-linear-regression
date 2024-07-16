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

def remove_outliers_zscore(df, z_threshold=2.5, percentile_threshold=0.01):
    # Calcular Z-scores
    z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
    abs_z_scores = np.abs(z_scores)

    # Filtrado por Z-score
    z_filtered = (abs_z_scores < z_threshold).all(axis=1)

    # Filtrado por percentiles
    lower_bound = df.quantile(percentile_threshold)
    upper_bound = df.quantile(1 - percentile_threshold)
    percentile_filtered = (df >= lower_bound) & (df <= upper_bound)

    # Combina ambos filtros
    combined_filter = z_filtered & percentile_filtered.all(axis=1)
    df_filtered = df[combined_filter]

    return df_filtered