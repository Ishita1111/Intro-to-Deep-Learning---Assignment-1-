import numpy as np


class NeuralLayer:
    def __init__(self, in_features, out_features, weight_init="random"):

        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        elif weight_init == "zeroes":
            self.W = np.zeros((in_features, out_features))
        else:  # random
            self.W = 0.01 * np.random.randn(in_features, out_features)
        
        self.b = np.zeros((1, out_features)) # bias shape should be (1, out_features)

        # gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input