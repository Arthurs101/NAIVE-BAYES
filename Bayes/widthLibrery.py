#Para realizar el mismo procedimiento pero con una libreria utilizaremos "sklearn"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

#Se utilizo pandas con la libreria "os" para leer el archivo desde el directorio actual 
df = pd.read_csv(os.path.join(os.getcwd(), "entrenamiento.txt"),delimiter="\t", header=None , names=["status","message"])

# Se dividio el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['status'], test_size=0.2, random_state=42)

# Se vectorizaron de los mensajes
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Naive Bayes con sklearn
model_sklearn = MultinomialNB()
model_sklearn.fit(X_train_vect, y_train)

# Predicciones
y_pred_train = model_sklearn.predict(X_train_vect)
y_pred_test = model_sklearn.predict(X_test_vect)

#Informe de clasificación
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)

# Imprimir resultados
print("Precisión en el conjunto de entrenamiento:", accuracy_train)
print("Precisión en el conjunto de prueba:", accuracy_test)
print("\nInforme de Clasificación:\n", report)