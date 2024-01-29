import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv("high_diamond_ranked_10min.csv")  
selected_features = ['blueGoldDiff', 'blueExperienceDiff']
X = data[selected_features]
y = data['blueWins']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo SVM con scikit-learn
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predicciones y evaluación
y_pred_sklearn = svm_classifier.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f'Accuracy (scikit-learn): {accuracy_sklearn}')


X_test_descaled = (X_test * scaler.scale_) + scaler.mean_
class_0_sklearn = X_test_descaled[y_test == 0]
class_1_sklearn = X_test_descaled[y_test == 1]

plt.figure(figsize=(10, 6))
plt.scatter(class_0_sklearn[:, 0], class_0_sklearn[:, 1], color='red', label='Clase 0 (No Gana Equipo Azul)')
plt.scatter(class_1_sklearn[:, 0], class_1_sklearn[:, 1], color='blue', label='Clase 1 (Gana Equipo Azul)')
plt.xlabel('Diferencia de Oro (blueGoldDiff)')
plt.ylabel('Diferencia de Experiencia (blueExperienceDiff)')
plt.legend()
plt.title('Visualización de los Grupos con SVM de Scikit-learn')
plt.show()
