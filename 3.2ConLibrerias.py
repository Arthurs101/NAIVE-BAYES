import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar los datos
data = pd.read_csv('high_diamond_ranked_10min.csv')

# Preparar los datos para el modelo
X = data.drop(columns='blueWins')  # Características
y = data['blueWins']               # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear el modelo de árbol de decisión
decision_tree = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
decision_tree.fit(X_train, y_train)

# Evaluar el modelo
y_pred_val = decision_tree.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f'Validation Accuracy: {val_accuracy}')

# Evaluar la importancia de las características
importances = decision_tree.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
top_features = feature_importance.sort_values(by='importance', ascending=False).head(5)
print('Top 5 features:', top_features)

# Evaluar con el conjunto de prueba
y_pred_test = decision_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Test Accuracy: {test_accuracy}')

# Técnicas para minimizar el overfitting
decision_tree_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)
decision_tree_tuned.fit(X_train, y_train)
y_pred_val_tuned = decision_tree_tuned.predict(X_val)
val_accuracy_tuned = accuracy_score(y_val, y_pred_val_tuned)
print(f'Tuned Validation Accuracy: {val_accuracy_tuned}')


