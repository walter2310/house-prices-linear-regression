# %%
import pandas as pd
from data_loader import train_test_split_data
import matplotlib.pylab as plt
from utils.data_cleaning import handle_missing_values, apply_log_transform, apply_one_hot_encoding
from utils.visualization import plot_histograms, plot_correlation_matrix
from sklearn.preprocessing import StandardScaler

def create_feature_engineering(df):
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/housing.csv')
    df = handle_missing_values(df)

    # Dividir datos en conjuntos de entrenamiento y prueba
    stratify_column = 'ocean_proximity'
    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split_data(df, stratify=stratify_column)

    # Combinar los datos de entrenamiento en un único set
    train_set = x_train_set.join(y_train_set)

    test_set = x_test_set.join(y_test_set)

    # Transformar valores categóricos en numéricos
    train_set = apply_one_hot_encoding(train_set, 'ocean_proximity')
    train_set = create_feature_engineering(train_set)
    log_transform_columns = ['total_rooms', 'total_bedrooms', 'population', 'households']
    train_set = apply_log_transform(train_set, log_transform_columns)

    test_set = apply_one_hot_encoding(test_set, 'ocean_proximity')
    test_set = create_feature_engineering(test_set)
    test_set = apply_log_transform(test_set, log_transform_columns)

    # Usando Standard Scaler para normalizar nuestro conjunto de datos
    scaler = StandardScaler()
    train_set = pd.DataFrame(scaler.fit_transform(train_set), columns=train_set.columns)
    test_set = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns)

    # Visualizar la matriz de correlación
    plot_correlation_matrix(train_set)

    # Visualizar histogramas de atributos seleccionados
    selected_attributes = ['total_rooms', 'total_bedrooms', 'population', 'households']
    plot_histograms(train_set, selected_attributes)

    train_set.to_csv('../data/processed/train_set.csv', index=False)
    test_set.to_csv('../data/processed/test_set.csv', index=False)