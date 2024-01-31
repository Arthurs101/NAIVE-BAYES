import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def calculate_gini(y):
    m = len(y)
    return 1 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

def best_split(X, y, num_classes, feature_importances):
    m, n = X.shape
    best_gini = 1
    best_idx, best_thr = None, None
    best_impurity_decrease = 0

    parent_gini = calculate_gini(y)

    for idx in range(n):
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * num_classes
        num_right = [np.sum(y == c) for c in range(num_classes)]
        for i in range(1, m):
            c = classes[i - 1]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1 - sum((num_left[x] / i) ** 2 for x in range(num_classes))
            gini_right = 1 - sum((num_right[x] / (m - i)) ** 2 for x in range(num_classes))
            gini = (i * gini_left + (m - i) * gini_right) / m
            impurity_decrease = parent_gini - gini
            if thresholds[i] == thresholds[i - 1]:
                continue
            if impurity_decrease > best_impurity_decrease:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                best_impurity_decrease = impurity_decrease

    if best_idx is not None:
        # Escalar la importancia de la característica por el número de muestras afectadas
        feature_importances[best_idx] += best_impurity_decrease * (m - sum(num_left if best_idx else num_right))

    return best_idx, best_thr



def build_tree(X, y, depth, max_depth, num_classes, feature_importances, min_samples_split=2, min_samples_leaf=1):
    num_samples = len(y)
    num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = DecisionTreeNode(
        gini=calculate_gini(y),
        num_samples=num_samples,
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
    )

    # Ver si es la  profundidad máxima o si se ha alcanzado el número mínimo de muestrar
    if depth < max_depth and num_samples >= min_samples_split:
        idx, thr = best_split(X, y, num_classes, feature_importances)
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]

            
            if len(y_left) >= min_samples_leaf and len(y_right) >= min_samples_leaf:
                node.feature_index = idx
                node.threshold = thr
                node.left, feature_importances = build_tree(X_left, y_left, depth + 1, max_depth, num_classes, feature_importances, min_samples_split, min_samples_leaf)
                node.right, feature_importances = build_tree(X_right, y_right, depth + 1, max_depth, num_classes, feature_importances, min_samples_split, min_samples_leaf)
    return node, feature_importances



def predict(node, sample):
    while node.left:
        if sample[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class

# Cargar y preparar el dataset
file_path = 'high_diamond_ranked_10min.csv'
data = pd.read_csv(file_path)

data = data.drop('gameId', axis = 1)
X = data.drop('blueWins', axis = 1)
X = X.values
y = data['blueWins'].values


# Dividir el dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Inicializar la importancia de las características
feature_importances = np.zeros(X.shape[1])

# Construir y entrenar el árbol
# Construcción y entrenamiento del árbol
max_depth = 5  
min_samples_split = 10  
min_samples_leaf = 5  
tree, feature_importances = build_tree(X_train, y_train, 0, max_depth, len(np.unique(y)), feature_importances, min_samples_split, min_samples_leaf)

# Hacer predicciones y evaluar el modelo
y_pred_val = [predict(tree, sample) for sample in X_val]
y_pred_test = [predict(tree, sample) for sample in X_test]

val_accuracy = np.mean(y_pred_val == y_val)
test_accuracy = np.mean(y_pred_test == y_test)

# Cálculo de la importancia relativa de las características
feature_importances /= np.sum(feature_importances)
sorted_indices = np.argsort(feature_importances)[::-1]
top_features = [(data.drop(columns='blueWins').columns[i], feature_importances[i]) for i in sorted_indices[:5]]

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
print("Top 5 Features:", top_features)