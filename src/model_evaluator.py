import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar el conjunto de prueba
TEST_SET_PATH = '../data/processed/test_set.csv'
df = pd.read_csv(TEST_SET_PATH)

x_test_set = df.drop("median_house_value", axis=1)
y_test_set = df["median_house_value"].copy()

# Cargar el modelo
MODEL_FILENAME = '../models/linear_regression_model.pkl'
lin_reg = joblib.load(MODEL_FILENAME)

# Hacer predicciones
y_pred = lin_reg.predict(x_test_set)

# Evaluar con métricas numéricas
mae = mean_absolute_error(y_test_set, y_pred)
mse = mean_squared_error(y_test_set, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_set, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Visualización: Diagrama de dispersión de predicciones vs. valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test_set, y_pred, alpha=0.3)
plt.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs. Predicciones')
plt.show()
