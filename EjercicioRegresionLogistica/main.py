import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns 

# Carga del conjunto de datos desde el archivo CSV
dataset = pd.read_csv("user+data.csv")

# Selecciona las columnas 2 y 3 como características (X) y la columna 4 como variable objetivo (y)
X = dataset.iloc[:, [2, 3]].values  # Columnas de características (edad, salario estimado)
y = dataset.iloc[:, 4].values       # Columna objetivo (compró o no el producto)

# Divide los datos en conjuntos de entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Estandarización de los datos para tener medias 0 y varianza 1 (normalización)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # Ajusta y transforma el conjunto de entrenamiento
X_test = sc_X.transform(X_test)        # Transforma el conjunto de prueba con el mismo escalador

# Creación y entrenamiento del modelo de regresión logística
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicción de los valores del conjunto de prueba
y_pred = classifier.predict(X_test)

# Generación de la matriz de confusión para evaluar el rendimiento del modelo
cm = confusion_matrix(y_pred, y_test)
print(cm)

# Visualización de la matriz de confusión usando un mapa de calor de Seaborn
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")  # Muestra los valores en cada celda
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
