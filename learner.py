import numpy as np

EPOCHS = 1000
ETA = 0.0005
np.random.seed(42)

def activation(x):
    return np.tanh(x)

def derivative(x):
    return 1 - np.tanh(x)**2

def shuffle(x, y):
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    return x[indices], y[indices]

class NeuralNetwork:
    def __init__(self):
        self.layers = [
            np.random.uniform(-np.sqrt(6 / 27), np.sqrt(6 / 27), (2, 25)),
            np.random.uniform(-np.sqrt(6 / 27), np.sqrt(6 / 27), (26, 1)),
        ]

    def forward(self, x):
        x = np.column_stack((x, np.ones(len(x))))
        self.hidden_sum = np.dot(x, self.layers[0])
        self.hidden_out = activation(self.hidden_sum)
        self.out = np.dot(
            np.column_stack((self.hidden_out, np.ones(len(self.hidden_out)))),
            self.layers[1]
        )
        return self.out


    def backward(self, x, y_true):
        out_error = self.out - y_true.reshape(-1, 1)
        hidden_error = np.dot(out_error, self.layers[1][:-1].T) * derivative(self.hidden_sum)
        self.layers[1] -= ETA * np.dot(
            np.column_stack((self.hidden_out, np.ones(len(self.hidden_out)))).T,
            out_error
        )
        self.layers[0] -= ETA * np.dot(
            np.column_stack((x, np.ones(len(x)))).T,
            hidden_error
        )

    def train(self, X, Y):
        normalized_X = (X - X.mean()) / (X.std() + 1e-5)
        self.normalization_X = (X.mean(), X.std())
        normalized_Y = (Y - Y.mean()) / (Y.std() + 1e-5)
        self.normalization_Y = (Y.mean(), Y.std())
        for i in range(EPOCHS):
            normalized_X, normalized_Y = shuffle(normalized_X, normalized_Y)
            self.forward(normalized_X)
            self.backward(normalized_X, normalized_Y)
            self.MSE = np.mean((normalized_Y - self.out.flatten()) ** 2)
            if (i + 1) % 100 == 0:
                print(f"Epoch {i + 1}: MSE = {self.MSE:.6f}")

    def predict(self, X):
        normalized_X = (X - self.normalization_X[0]) / (self.normalization_X[1] + 1e-5)
        self.forward(normalized_X)
        return self.out * self.normalization_Y[1] + self.normalization_Y[0]
