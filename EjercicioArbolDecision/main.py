import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

# Definir los nombres de las columnas para el DataFrame
col_names = ['company', 'job', 'degree', 'salary_more_then_100k']

# Cargar el archivo CSV con los datos
data = pd.read_csv('salaries.csv')

# Definir las columnas de características (variables independientes)
feature_cols = ['company', 'job', 'degree']
X = data[feature_cols]  # Asigna las características a X
y = data['salary_more_then_100k']  # La variable objetivo (salarios > 100k) se asigna a y

# Inicializar el codificador de etiquetas para convertir variables categóricas en numéricas
label_encoder = preprocessing.LabelEncoder()

# Convertir las columnas categóricas a valores numéricos con LabelEncoder
data["company"] = label_encoder.fit_transform(data["company"])
data["job"] = label_encoder.fit_transform(data["job"])
data["degree"] = label_encoder.fit_transform(data["degree"])

# Redefinir las columnas de características ahora que están codificadas
feature_cols = ["company", "job", "degree"]
X = data[feature_cols]  # Asigna las características transformadas a X
y = data["salary_more_then_100k"]  # Variable objetivo sigue siendo la misma

# Dividir el conjunto de datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Crear el clasificador de Árbol de Decisión utilizando el criterio de entropía y una profundidad máxima de 2
clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=2)

# Entrenar el modelo con los datos de entrenamiento
clf_entropy = clf_entropy.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = clf_entropy.predict(X_test)

# Evaluar la precisión del modelo
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
