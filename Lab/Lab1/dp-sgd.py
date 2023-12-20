import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
class LogLevel:
    DEBUG = 0
    INFO = 1 
    WARNING = 2
    ERROR = 3
    FATAL = 4
LOG_LEVEL = LogLevel.INFO

def clip(x, lower_bound, upper_bound):
    return np.minimum(np.maximum(x, lower_bound), upper_bound)

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tau = 1e-6  # small value to prevent log(0)
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # clip(z, -300, 300)

    def __fit(self, X, y, is_dp=False, epsilon=None, delta=None, C=1):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        # kaiming_uniform_bound = np.sqrt(3.0) / np.sqrt(num_features) # kaiming_uniform
        # self.weights = np.random.uniform(-kaiming_uniform_bound, kaiming_uniform_bound, size=(num_features,))
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Gradient descent optimization
        for epoch in range(self.num_iterations):
            # Compute predictions of the model
            predictions = self.predict_probability(X)
            # predictions.shape = (455,)

            # Compute loss and gradients
            loss = -np.mean(
                y * np.log(predictions + self.tau)
                + (1 - y) * np.log(1 - predictions + self.tau)
            ) # Cross entropy loss
            
            dz = -(y / (predictions + self.tau) - (1 - y) / (1 - predictions + self.tau)) # Cross entropy loss
            # dz = predictions - y # L2_loss
            dz = dz * (predictions * (1 - predictions)) # sigmoid derivative

            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            if LOG_LEVEL <= LogLevel.DEBUG and epoch % 200 == 199:
                print("Epoch:", epoch, "Loss:", loss)
                print("dw:", dw)
                print("db:", db)
                print("weights:", self.weights)
                print("bias:", self.bias)
                print("predictions[:10]:", predictions[:10])
                print("y[:10]:", y[:10])

            if is_dp:
                assert epsilon is not None and delta is not None
                dw = X * dz.reshape(-1, 1) # grad of every sample
                # Clip gradient
                dw = clip_gradients(dw, C)
                # Add noise to gradients
                # *-TODO: Calculate epsilon_u, delta_u based epsilon, delta and epochs here.
                # 由Advanced Composition Theorem可知，对于每个epoch，我们可以将epsilon和delta分别除以epoch数，然后加到每个epoch上，这样就可以保证总的epsilon和delta不变。
                k = self.num_iterations
                # epsilon_u, delta_u = epsilon / k, delta / k
                delta_u = delta / (k + 1)
                # 近似计算epsilon_u
                epsilon_u = epsilon / np.sqrt(2 * k * np.log(1 / delta_u))
                # 下面式子中，分母中的epsilon_u大于真实的epsilon_u，因此可以保证epsilon_u的计算结果偏小，符合隐私保证
                epsilon_u = epsilon / (np.sqrt(2 * k * np.log(1 / delta_u)) + k * (np.exp(epsilon_u) - 1)) 
                dw = add_gaussian_noise_to_gradients(dw, epsilon_u, delta_u, C)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def fit(self, X, y):
        self.__fit(X, y)

    def dp_fit(self, X, y, epsilon, delta, C):
        self.__fit(X, y, is_dp=True, epsilon=epsilon, delta=delta, C=C)
        # 计算花费的隐私预算
        k = self.num_iterations
        # epsilon_u, delta_u = epsilon / k, delta / k
        delta_u = delta / (k + 1)
        # 近似计算epsilon_u
        epsilon_u = epsilon / np.sqrt(2 * k * np.log(1 / delta_u))
        # 下面式子中，分母中的epsilon_u大于真实的epsilon_u，因此可以保证epsilon_u的计算结果偏小，符合隐私保证
        epsilon_u = epsilon / (np.sqrt(2 * k * np.log(1 / delta_u)) + k * (np.exp(epsilon_u) - 1))
        epsilon_cost = np.sqrt(2 * k * np.log(1 / delta_u)) * epsilon_u + k * epsilon_u * (np.exp(epsilon_u) - 1)
        delta_cost = delta_u * k + delta_u
        print("epsilon_cost:", epsilon_cost)
        print("delta_cost:", delta_cost)

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
        np.random.seed(RANDOM_STATE) # 种子，保证结果可复现
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=RANDOM_STATE
        )
        """
        生成一个随机的n类分类问题。
        其样本点初始化为围绕n_informative维超立方体顶点（边长为2 * class_sep = 1）正态分布（std = 1）的点的簇，并将相等数量的簇分配给每个类。
        每个输入为n_feature维，其中n_informative维特征是有信息的，n_redundant维为特征的线性组合，n_repeated维为随机噪声，其余维度随机抽取前面维度数据填充。
        """
    elif dataset_name == "cancer":
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        # input_dim: 30   output_dim: 1  num_samples: 569   split: [455, 114]
    else:
        raise ValueError("Not supported dataset_name.")

    # normalize the data
    X = (X - np.mean(X, axis=0)) / X.std(axis=0)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    if LOG_LEVEL <= LogLevel.DEBUG:
        print("X_train.shape:", X_train.shape) 
        print("X_test.shape:", X_test.shape)
        print("y_train.shape:", y_train.shape)
        print("y_test.shape:", y_test.shape)
        print("X_test[:3]:", X_test[:3])
        print("y_test[:3]:", y_test[:3])

    return X_train, X_test, y_train, y_test


def clip_gradients(sample_gradients, C):
    # *-TODO: Clip gradients.
    grad_norm = np.linalg.norm(sample_gradients, ord=2, axis=1)
    clip_factor = np.maximum(1, grad_norm/C)
    clip_gradients = sample_gradients / clip_factor[:, np.newaxis]
    return clip_gradients


def add_gaussian_noise_to_gradients(sample_gradients, epsilon, delta, C):
    # *-TODO: add gaussian noise to gradients.
    gradients = np.sum(sample_gradients, axis=0)

    std = C * np.sqrt(2 * np.log(1.25 / delta)) / epsilon # 上述梯度裁剪已经保证了C即为sensitivity的上界
    noise = np.random.normal(loc=0.0, scale=std, size=gradients.shape)
    noisy_gradients = gradients + noise

    noisy_gradients = noisy_gradients / sample_gradients.shape[0]

    return noisy_gradients


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE) # 种子，保证结果可复现
    # Prepare datasets.
    dataset_name = "cancer"
    X_train, X_test, y_train, y_test = get_train_data(dataset_name)

    LEARNING_RATE = 0.01
    NUM_ITERATIONS = 100

    # Training the normal model
    normal_model = LogisticRegressionCustom(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    normal_model.fit(X_train, y_train)
    y_pred = normal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Normal accuracy:", accuracy)

    # Training the differentially private model
    dp_model = LogisticRegressionCustom(learning_rate=LEARNING_RATE, num_iterations=NUM_ITERATIONS)
    epsilon, delta = 1.0, 1e-3
    dp_model.dp_fit(X_train, y_train, epsilon=epsilon, delta=delta, C=1)
    y_pred = dp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("DP accuracy:", accuracy)
