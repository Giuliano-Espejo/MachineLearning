import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa herramientas para gráficos 3D
from sklearn.cluster import KMeans        # Importa el algoritmo K-Means para clustering
from sklearn.datasets import make_blobs   # Importa una función para crear datos de prueba

# Configura el tamaño de la figura de los gráficos
plt.rcParams["figure.figsize"] = (16, 9)

# Genera un conjunto de datos sintético con 800 muestras, 3 características y 4 centros
X, y = make_blobs(n_samples=800, n_features=3, centers=4)

# Lista para almacenar el valor de WCSS (Within-Cluster Sum of Squares) para cada número de clústeres
wcss_list = []

# Loop para calcular WCSS para diferentes cantidades de clústeres (de 1 a 10)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=4)  # Inicializa el modelo K-Means
    kmeans.fit(X)  # Ajusta el modelo a los datos
    wcss_list.append(kmeans.inertia_)  # Almacena el valor de WCSS

# Gráfico del método del codo (Elbow Method) para determinar el número óptimo de clústeres
plt.plot(range(1, 11), wcss_list)
plt.title("The Elbow Method")  # Título del gráfico
plt.xlabel("Number of Clusters")  # Etiqueta del eje X
plt.ylabel("WCSS")  # Etiqueta del eje Y
plt.show()  # Muestra el gráfico
