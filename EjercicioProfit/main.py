import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataSet = pd.read_csv("50_Startups.csv")

#x contiene todas las características o variables independientes (entrada) del conjunto de datos, excluyendo la última columna
X = dataSet.iloc[:,:-1]

#y contiene la variable dependiente (objetivo) que usualmente es la última columna del conjunto de datos.
#Convierte los datos seleccionados en un array de NumPy. Esto es útil cuando trabajas con librerías como scikit-learn, 
  #que esperan los datos en forma de arrays en lugar de DataFrames de pandas.
y= dataSet.iloc[:, -1].values

#muestra los valores, se le puede pasar la cantidad de filas que se desea mostra
print(dataSet.head())

#ColumnTransformer: Esta clase permite aplicar diferentes transformaciones a diferentes columnas de tu conjunto
 # de datos. Aquí se utiliza para aplicar One-Hot Encoding solo a una columna específica.
from sklearn.compose import ColumnTransformer

#OneHotEncoder: Este es el codificador utilizado para transformar una columna categórica en múltiples columnas binarias. 
 # Cada categoría en la columna original se convierte en una columna independiente con valores 0 o 1, lo que es útil para el modelado en machine learning.
from sklearn.preprocessing import OneHotEncoder


#transformers=[...]: Esta lista define las transformaciones que se van a aplicar a ciertas columnas. Aquí está usando One-Hot Encoding en la columna que se
  #  encuentra en el índice 3 (la cuarta columna de X).
#'encoder': Nombre del transformador. Este es solo un identificador arbitrario.
#OneHotEncoder(): El codificador que se aplicará a la columna categórica.
#[3]: La columna a la que se aplicará el codificador One-Hot. En este caso, es la columna con índice 3 (recuerda que los índices en Python empiezan desde 0).
#remainder='passthrough': Esto indica que el resto de las columnas que no se transforman deben ser "pasadas" o dejadas sin cambios. Es decir, solo la columna 
  # en el índice 3 será transformada, mientras que las demás columnas se mantendrán tal como están.
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough')

#fit_transform: Ajusta (aprende) la transformación a la columna seleccionada y luego la transforma.
#X: Es el conjunto de características que se va a transformar. La columna categórica en el índice 3 será transformada mediante One-Hot Encoding.
#np.array(...): Convierte el resultado en un array de NumPy. Esto es necesario porque el método fit_transform devuelve un DataFrame si estás 
  # trabajando con pandas, y en muchos casos es preferible trabajar con arrays de NumPy, especialmente para ciertas librerías de machine learning.
X = np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split

#X: Conjunto de características (variables independientes).
#y: Conjunto de etiquetas o valores a predecir (variable dependiente).
#train_test_split(): Divide los datos en conjuntos de entrenamiento y prueba.
#test_size=0.2: El 20% de los datos se utilizarán para pruebas.
#X_train y y_train: Datos de entrenamiento.
#X_test y y_test: Datos de prueba.
#Esta división es clave para evaluar el rendimiento del modelo de machine learning en datos no vistos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)


from sklearn.linear_model import LinearRegression
#Esta línea crea una instancia de un modelo de regresión lineal de la biblioteca sklearn
regressor = LinearRegression()
#Esta línea entrena el modelo de regresión lineal con los datos proporcionados
regressor.fit(X_train, y_train)

#Esta línea utiliza el modelo entrenado para predecir el precio de una casa con un área de 3300 pies cuadrados. 
y_pred = regressor.predict(X_test)

#Esta línea de código crea un DataFrame de pandas que contiene dos columnas: los valores reales y los valores predichos.
#  Esto es útil para comparar cómo de bien tu modelo de machine learning está prediciendo los valores.
#y_test: Son los valores reales del conjunto de prueba, es decir, los valores correctos que debería predecir el modelo.
#y_pred: Son los valores predichos por el modelo para los datos de prueba.
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print()
print(df)


"""
-------------------------------
---- ERROR CUADRATICO MEDIO----
-------------------------------
"""

from sklearn import metrics

"""
metrics.mean_squared_error(y_test, y_pred): Calcula el error cuadrático medio (MSE) entre los valores reales (y_test)
y los valores predichos (y_pred). El MSE mide el promedio de los errores al cuadrado, lo que amplifica grandes errores,
penalizando más los valores que están lejos de la predicción.

np.sqrt(...): Aplica la raíz cuadrada al MSE para obtener el error cuadrático medio raíz (RMSE). El RMSE devuelve los
errores en las mismas unidades que la variable dependiente (en lugar de las unidades al cuadrado).
"""
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #8953.194096341342

#El valor 8953.19 que arroja el programa corresponde a la raíz cuadrada del error cuadrático medio (RMSE),
#  calculado entre los valores reales (y_test) y los valores predichos (y_pred).

#El RMSE mide la magnitud promedio de los errores entre las predicciones del modelo y los valores reales. 
# Un valor de 8953.19 indica que, en promedio, el modelo se está equivocando en aproximadamente 8953 unidades al hacer sus predicciones.

#Un valor alto de RMSE sugiere que el modelo no está prediciendo con mucha precisión.
