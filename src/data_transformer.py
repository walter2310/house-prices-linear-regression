import pandas as pd
from data_loader import train_test_split_data, check_data
import matplotlib.pylab as plt
from utils.data_cleaning import handle_missing_values, apply_log_transform, apply_one_hot_encoding, remove_outliers_zscore
from utils.visualization import plot_histograms, plot_correlation_matrix, box_plot
from sklearn.preprocessing import StandardScaler

def create_feature_engineering(df):
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/housing.csv')
    df = handle_missing_values(df)
    stratify_column = 'ocean_proximity'

    # Dividir datos en conjuntos de entrenamiento y prueba
    x_train_set, x_test_set, y_train_set, y_test_set = train_test_split_data(df, stratify=stratify_column)

    # Aplicar transformaciones solo a las características de entrenamiento
    x_train_set = apply_one_hot_encoding(x_train_set, 'ocean_proximity')
    x_train_set = create_feature_engineering(x_train_set)
    log_transform_columns = ['total_rooms', 'total_bedrooms', 'population', 'households', 'bedroom_ratio', 'household_rooms']
    x_train_set = apply_log_transform(x_train_set, log_transform_columns)

    # Usando Standard Scaler para normalizar nuestro conjunto de datos
    scaler = StandardScaler()
    x_train_set_scaled = pd.DataFrame(scaler.fit_transform(x_train_set), columns=x_train_set.columns)

    check_data(x_train_set_scaled)
    x_train_set_scaled = remove_outliers_zscore(x_train_set_scaled)

    x_test_set = apply_one_hot_encoding(x_test_set, 'ocean_proximity')
    x_test_set = create_feature_engineering(x_test_set)
    x_test_set = apply_log_transform(x_test_set, log_transform_columns)
    x_test_set_scaled = pd.DataFrame(scaler.transform(x_test_set), columns=x_test_set.columns)

    x_test_set_scaled = remove_outliers_zscore(x_test_set_scaled)

    # Combinar los datos de entrenamiento y prueba en sus respectivos sets
    train_set = x_train_set_scaled.join(y_train_set.reset_index(drop=True))
    test_set = x_test_set_scaled.join(y_test_set.reset_index(drop=True))

    # Visualizar la matriz de correlación (opcional)
    # plot_correlation_matrix(train_set)

    # Visualizar histogramas de atributos seleccionados (opcional)
    selected_attributes = ['total_rooms', 'total_bedrooms', 'population', 'households', 'bedroom_ratio', 'household_rooms']
    # plot_histograms(train_set, selected_attributes)

    check_data(train_set)

    train_set.to_csv('../data/processed/train_set.csv', index=False)
    test_set.to_csv('../data/processed/test_set.csv', index=False)
