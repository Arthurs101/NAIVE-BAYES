import numpy as np

filename = "C:\\Users\\HP OMEN\\OneDrive\\Desktop\\Programs\\AI\\Lab2high_diamond_ranked_10min.csv"

# Load data from CSV
def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, 1:]  # Assuming your data starts from the second column
    y = data[:, 0]   # Assuming the first column is the target variable

    # Convert y to binary (0 or 1)
    y = (y > 0).astype(int)

    return X, y

# SVM class definition
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for epoch in range(self.num_epochs):
            # Compute SVM loss and gradients
            loss, dw, db = self.compute_loss_and_gradients(X, y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def compute_loss_and_gradients(self, X, y):
        num_samples = X.shape[0]

        # Compute SVM loss
        scores = np.dot(X, self.weights) + self.bias
        margins = 1 - y * scores
        margins = np.maximum(0, margins)
        loss = np.sum(margins) / num_samples + 0.5 * self.lambda_param * np.sum(self.weights**2)

        # Compute gradients
        margins[margins > 0] = 1
        incorrect_classifications = np.sum(margins, axis=0)
        margins[y.astype(int)] -= incorrect_classifications
        dw = np.dot(X.T, margins) / num_samples + self.lambda_param * self.weights
        db = np.sum(margins) / num_samples

        return loss, dw, db

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.sign(scores)

# Main part
if __name__ == "__main__":
    # Load data
    filename = "high_diamond_ranked_10min.csv"
    X, y = load_data(filename)

    # Initialize and train SVM
    svm = SVM()
    svm.train(X, y)

    # Make predictions
    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")
