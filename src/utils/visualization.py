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

def box_plot(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot of {column}')
        plt.show()
