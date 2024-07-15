import matplotlib.pylab as plt
import seaborn as sns

# Función para visualizar la matriz de correlación
def plot_correlation_matrix(df):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
    plt.title('Matriz de Correlación')
    plt.show()


# Función para visualizar histogramas de atributos seleccionados
def plot_histograms(df, attributes):
    df[attributes].hist(figsize=(15, 8), bins=50, edgecolor='black')
    plt.suptitle('Histograma de Atributos Transformados')
    plt.show()
