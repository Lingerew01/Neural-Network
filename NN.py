import numpy as np


class Net:
    def __init__(self, layer_sizes, activation="sigmoid"):
        """
        Initialize the neural network with the given layer sizes and activation function.

        """
        self.weights = []
        self.biases = []
        self.outputs = []
        self.activation = activation

        # Initialize weights, biases, and optimizer-specific parameters
        self.momentum = []
        self.rms = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(np.float32))
            self.biases.append(np.zeros((1, layer_sizes[i + 1]), dtype=np.float32))
            self.momentum.append(np.zeros((layer_sizes[i], layer_sizes[i + 1]), dtype=np.float32))
            self.rms.append(np.zeros((layer_sizes[i], layer_sizes[i + 1]), dtype=np.float32))
        self.adam_t = 1  # Time step for Adam optimizer

    def activate(self, x):
        """
        Apply the chosen activation function.
        """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def grad_activate(self, x):
        """

        """
        if self.activation == "sigmoid":
            return x * (1 - x)
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)
        elif self.activation == "tanh":
            return 1 - np.square(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def forward(self, x):
        """
        Perform a forward pass through the network.

        """
        if x.shape[1] != self.weights[0].shape[0]:
            raise ValueError(f"Input size {x.shape[1]} does not match expected size {self.weights[0].shape[0]}")

        self.outputs = [x]
        for w, b in zip(self.weights, self.biases):
            x = self.activate(np.dot(x, w) + b)
            self.outputs.append(x)
        return x

    def compute_loss(self, y_true, y_pred, loss="mse"):
        """
        Compute loss and its gradient.

        """
        if loss == "mse":
            return np.mean((y_true - y_pred) ** 2), -(y_true - y_pred)
        elif loss == "cross_entropy":
            epsilon = 1e-8
            return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)), \
                (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

    def backpropagation(self, y_true, learning_rate=0.01, optimizer="SGD", loss="mse", beta1=0.9, beta2=0.999,
                        epsilon=1e-8):
        """
        Perform backpropagation and update weights and biases.

        """
        # Compute loss and initial error
        loss_value, error = self.compute_loss(y_true, self.outputs[-1], loss=loss)
        delta = error * self.grad_activate(self.outputs[-1])

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grad_weights = np.dot(self.outputs[i].T, delta)
            grad_biases = np.sum(delta, axis=0, keepdims=True)

            if optimizer == "SGD":
                self.weights[i] -= learning_rate * grad_weights
                self.biases[i] -= learning_rate * grad_biases
            elif optimizer == "Adam":
                self.momentum[i] = beta1 * self.momentum[i] + (1 - beta1) * grad_weights
                self.rms[i] = beta2 * self.rms[i] + (1 - beta2) * (grad_weights ** 2)
                m_hat = self.momentum[i] / (1 - beta1 ** self.adam_t)
                v_hat = self.rms[i] / (1 - beta2 ** self.adam_t)
                self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                self.biases[i] -= learning_rate * grad_biases

            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.grad_activate(self.outputs[i])

        if optimizer == "Adam":
            self.adam_t += 1

        return loss_value







