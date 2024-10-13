# Importación de bibliotecas necesarias
import numpy as np  # Para operaciones numéricas
import pandas as pd  # Para manejo de datos
import matplotlib.pyplot as plt  # Para visualización de gráficos
import seaborn as sns  # Para visualización estadística mejorada

# Configuración del estilo de Seaborn
sns.set_style("white")

# Definición de nombres de las columnas para el conjunto de datos
column_names = ["user_id", "item_id", "rating", "timestamp"]

# Carga del conjunto de datos de ratings desde un archivo
df = pd.read_csv("u.data", sep="\t", names=column_names)
#print(df.head())  # Imprimir las primeras filas del DataFrame

# Carga de títulos de películas
movi_titles = pd.read_csv("Movie_Id_Titles")
#print(movi_titles.head())  # Imprimir las primeras filas del DataFrame de títulos

# Unión de los DataFrames de ratings y títulos de películas
df = pd.merge(df, movi_titles, on="item_id")
#print(df.head())  # Imprimir las primeras filas del DataFrame combinado

# Agrupación por título de película para contar y calcular la media de ratings
#print(df.groupby("title")["rating"].count().sort_values(ascending=False).head())
#print(df.groupby("title")["rating"].mean().sort_values(ascending=False).head())

# Creación de un DataFrame que contiene la media de ratings por película
ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
#print(ratings.head())  # Imprimir las primeras filas del DataFrame de ratings

# Adición de la cantidad de ratings por película
ratings["num of ratings"] = pd.DataFrame(df.groupby("title")["rating"].count())
#print(ratings.head())  # Imprimir el DataFrame con la nueva columna

# Visualización de la distribución del número de ratings
plt.figure(figsize=(10, 4))
ratings["num of ratings"].hist(bins=70)  # Histograma del número de ratings
# plt.show()  # Mostrar el gráfico (comentado para no mostrar todavía)

# Visualización de la distribución de las medias de ratings
plt.figure(figsize=(10, 4))
ratings["rating"].hist(bins=70)  # Histograma de los ratings
# plt.show()  # Mostrar el gráfico (comentado para no mostrar todavía)

# Creación de un gráfico conjunto para mostrar la relación entre ratings y número de ratings
sns.jointplot(x="rating", y="num of ratings", data=ratings, alpha=0.5)

# Creación de una matriz de datos donde las filas son usuarios y las columnas son títulos de películas
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
#print(moviemat.head())  # Imprimir las primeras filas de la matriz de películas

#print(ratings.sort_values('num of ratings', ascending=False).head(10))  # Imprimir las 10 películas con más ratings

# Extracción de ratings para películas específicas
starwars_user_ratings = moviemat['Star Wars (1977)']  # Ratings para "Star Wars"
liarliar_user_ratings = moviemat['Liar Liar (1997)']  # Ratings para "Liar Liar"
#print(starwars_user_ratings.head())  # Imprimir los ratings de "Star Wars"

# Cálculo de la correlación de ratings entre "Star Wars" y todas las demás películas
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)  # Correlación para "Liar Liar"

# Creación de DataFrames para las correlaciones
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)  # Eliminación de valores NaN
#print(corr_starwars.head())  # Imprimir las correlaciones de "Star Wars"

#print(corr_starwars.sort_values('Correlation', ascending=False).head(10))  # Imprimir las 10 películas más similares a "Star Wars"

# Adición del número de ratings al DataFrame de correlaciones
corr_starwars = corr_starwars.join(ratings['num of ratings'])
#print(corr_starwars.head())  # Imprimir el DataFrame con el número de ratings

#print(corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())  # Películas similares a "Star Wars" con más de 100 ratings

# Repetición del proceso para "Liar Liar"
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)  # Eliminación de valores NaN
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])  # Adición del número de ratings
#print(corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending=False).head())  # Películas similares a "Liar Liar" con más de 100 ratings
