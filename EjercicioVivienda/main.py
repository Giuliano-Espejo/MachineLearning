import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Carga los datos en una variable
df = pd.read_csv("homeprices.csv")

#Crea un grafico lineal
#titulo eje x
plt.xlabel('Area(sqr ft)')
#titulo eje y
plt.ylabel('Price(US$)')
#color es para el color de los puntos y marker para la forma que van a tener los puntos df.Area y df.Price son los valores
plt.scatter(df.Area, df.Price, color = 'red', marker='+')

#Muestra el grafico
#plt.show()

#Esta línea crea una instancia de un modelo de regresión lineal de la biblioteca sklearn
reg = LinearRegression()
#Esta línea entrena el modelo de regresión lineal con los datos proporcionados
reg.fit(df[['Area']], df.Price)


#Esta línea utiliza el modelo entrenado para predecir el precio de una casa con un área de 3300 pies cuadrados. 
print(reg.predict([[3300]]))

#Esta línea imprime el coeficiente (pendiente) del modelo ajustado. En un modelo de regresión lineal simple,
#este valor indica cómo cambia el precio de la vivienda con respecto al área. Es decir, por cada incremento 
#de 1 pie cuadrado en el área, el precio de la vivienda cambia en la cantidad dada por el coeficiente.
print(reg.coef_)

# Esta línea realiza el cálculo manual de la predicción para una casa de 3300 pies cuadrados utilizando la ecuación de una línea recta
print(135.788 * 3300 + 180616.438)