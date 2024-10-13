# Importación de bibliotecas necesarias
from copy import deepcopy  # Copia profunda (no se usa en este código)
import numpy as np  # Biblioteca para operaciones numéricas
import pandas as pd  # Biblioteca para manejo de datos
from matplotlib import pyplot as plt  # Biblioteca para visualización
from sklearn.cluster import KMeans  # Algoritmo de agrupamiento KMeans
from mpl_toolkits.mplot3d import Axes3D  # Herramientas para gráficos 3D

# Configuración del tamaño de la figura y el estilo
plt.rcParams["figure.figsize"] = (16, 9)
plt.style.use("ggplot")  # Estilo de gráfico de ggplot

# Carga de datos desde un archivo CSV
data = pd.read_csv("mallCustomerData.txt", sep=",")
#print(data.shape)  # Imprimir la forma del DataFrame
#print(data.head(10))  # Imprimir las primeras 10 filas del DataFrame

#print(data["Gender"].value_counts())  # Contar el número de ocurrencias de cada género

# Extracción de características de interés
f1 = data["Annual Income (k$)"].values  # Ingresos anuales
f2 = data["Spending Score (1-100)"].values  # Puntaje de gasto

# Imprimir las claves del DataFrame
for key in data.keys():
    print(key)

# Creación de un arreglo 2D combinando las dos características
X = np.array(list(zip(f1, f2)))

# Visualización de los datos en un gráfico de dispersión
plt.scatter(f1, f2, c="black", s=20)  # Graficar puntos en negro
# plt.show()  # Mostrar el gráfico (comentado para no mostrar todavía)

# Aplicación del algoritmo KMeans con 3 clústeres
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)  # Entrenamiento del modelo KMeans

# Predicción de las etiquetas de clúster para los datos
labels = kmeans.predict(X)
C = kmeans.cluster_centers_  # Obtención de los centros de los clústeres

# Creación de una figura para visualización 3D
fig = plt.figure()
ax = Axes3D(fig)  # Inicialización del objeto de gráfico 3D
# Graficar los datos originales en 3D
ax.scatter(X[:, 0], X[:, 1], X[:, 1], c="y")  # Puntos en amarillo
# Graficar los centros de los clústeres
ax.scatter(C[:, 0], C[:, 1], C[:, 1], marker='*', c='#050505', s=20)  # Estrella negra en los centros
plt.show()  # Mostrar el gráfico

# Aplicación de KMeans nuevamente con 4 clústeres
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)  # Entrenamiento del modelo KMeans
labels = kmeans.predict(X)  # Predicción de las etiquetas de clúster
C = kmeans.cluster_centers_  # Obtención de los nuevos centros de los clústeres
