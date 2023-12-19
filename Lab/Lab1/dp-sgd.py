import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tau = 1e-6  # small value to prevent log(0)
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Compute predictions of the model
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute loss and gradients
            loss = -np.mean(
                y * np.log(predictions + self.tau)
                + (1 - y) * np.log(1 - predictions + self.tau)
            )
            dz = predictions - y
            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def dp_fit(self, X, y, epsilon, delta, C=1):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            # Compute predictions of the model
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute loss and gradients
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            dz = predictions - y
            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            # Clip gradient
            clip_gradients = clip_gradients(dw, C)
            # Add noise to gradients
            # TODO: Calculate epsilon_u, delta_u based epsilon, delta and epochs here.
            epsilon_u, delta_u = None, None
            noisy_dw = add_gaussian_noise_to_gradients(dw, epsilon_u, delta_u, C)

            # Update weights and bias
            self.weights -= self.learning_rate * noisy_dw
            self.bias -= self.learning_rate * db

    def predict_probability(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_probability(X)
        # Convert probabilities to classes
        return np.round(probabilities)


def get_train_data(dataset_name=None):
    if dataset_name is None:
        # Generate simulated data
        np.random.seed(RANDOM_STATE)
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=RANDOM_STATE
        )
    elif dataset_name == "cancer":
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError("Not supported dataset_name.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def clip_gradients(gradients, C):
    # TODO: Clip gradients.
    clip_gradients = None
    return clip_gradients


def add_gaussian_noise_to_gradients(gradients, epsilon, delta, C):
    # TODO: add gaussian noise to gradients.
    noisy_gradients = None
    return noisy_gradients


if __name__ == "__main__":
    # Prepare datasets.
    dataset_name = "cancer"
    X_train, X_test, y_train, y_test = get_train_data(dataset_name)

    # Training the normal model
    normal_model = LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000)
    normal_model.fit(X_train, y_train)
    y_pred = normal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Normal accuracy:", accuracy)

    # Training the differentially private model
    dp_model = LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000)
    epsilon, delta = 1.0, 1e-3
    dp_model.dp_fit(X_train, y_train, epsilon=epsilon, delta=delta, C=1)
    y_pred = normal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("DP accuracy:", accuracy)
