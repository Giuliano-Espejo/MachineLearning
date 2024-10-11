import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Carga del conjunto de datos desde un archivo CSV
dataset = pd.read_csv("user+data.csv") 

# Separación de características (X) y la variable objetivo (y)
X = dataset.iloc[:, 2:4].values   # Selecciona las columnas 2 y 3 para las características
y = dataset.iloc[:, 4].values     # Selecciona la columna 4 para la variable objetivo

# División del conjunto de datos en entrenamiento y prueba (75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalización de los datos: se ajusta el escalador con X_train y se transforma X_train y X_test
sc_X = StandardScaler()           # Crea una instancia de StandardScaler para normalizar los datos
X_train = sc_X.fit_transform(X_train)   # Ajusta el escalador a X_train y luego lo transforma
X_test = sc_X.transform(X_test)         # Transforma X_test usando el escalador ajustado con X_train

# Creación del modelo K-Nearest Neighbors (KNN) con 5 vecinos y la métrica de Minkowski (distancia Euclidiana)
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)

# Entrenamiento del modelo KNN con los datos de entrenamiento
classifier.fit(X_train, y_train)

# Predicción de las etiquetas del conjunto de prueba
y_pred = classifier.predict(X_test)

# Creación de la matriz de confusión para evaluar el rendimiento del modelo
cm = confusion_matrix(y_pred, y_test)
print(cm)  # Imprime la matriz de confusión
