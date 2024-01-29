import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Implementación de SVM 
class BalancedSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Cargar y preparar los datos
data = pd.read_csv("high_diamond_ranked_10min.csv")  
selected_features = ['blueGoldDiff', 'blueExperienceDiff']  
X = data[selected_features]
y = data['blueWins']

# Escalar las características
X_scaled = (X - X.mean()) / X.std()

# División manual del dataset en entrenamiento, validación y prueba
np.random.seed(42)  # Para reproducibilidad
shuffle_indices = np.random.permutation(np.arange(len(X_scaled)))
X_shuffled = X_scaled.iloc[shuffle_indices]
y_shuffled = y.iloc[shuffle_indices]

train_size = int(len(X_shuffled) * 0.8)
val_size = int(len(X_shuffled) * 0.1)
test_size = len(X_shuffled) - train_size - val_size

X_train = X_shuffled.iloc[:train_size]
y_train = y_shuffled.iloc[:train_size]
X_val = X_shuffled.iloc[train_size:train_size + val_size]
y_val = y_shuffled.iloc[train_size:train_size + val_size]
X_test = X_shuffled.iloc[train_size + val_size:]
y_test = y_shuffled.iloc[train_size + val_size:]

# Crear y entrenar el modelo SVM
svm_model = BalancedSVM()
svm_model.fit(X_train.values, y_train.values)

# Predicciones y evaluación en el conjunto de prueba
y_pred = svm_model.predict(X_test.values)
accuracy = np.mean(y_pred == y_test.values)
print(f'Accuracy: {accuracy}')

# Visualización
class_0 = X_test[y_test == 0]
class_1 = X_test[y_test == 1]

plt.figure(figsize=(10, 6))
plt.scatter(class_0.iloc[:, 0], class_0.iloc[:, 1], color='red', label='Clase 0 (No Gana Equipo Azul)')
plt.scatter(class_1.iloc[:, 0], class_1.iloc[:, 1], color='blue', label='Clase 1 (Gana Equipo Azul)')
plt.xlabel('Diferencia de Oro (blueGoldDiff)')
plt.ylabel('Diferencia de Experiencia (blueExperienceDiff)')
plt.legend()
plt.title('Visualización de los Grupos con SVM Implementado Desde Cero')
plt.show()
