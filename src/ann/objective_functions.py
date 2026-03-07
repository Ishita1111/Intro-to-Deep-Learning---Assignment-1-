import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, logits, y_true):
        n = y_true.shape[0]
        grad = (2.0 / n) * (logits - y_true)
        return grad


class CrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None
        self.eps = 1e-12

    def forward(self, logits, y_true):
        self.y_true = y_true

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        batch_size = logits.shape[0]
        loss = -np.mean(
            np.log(self.probs[np.arange(batch_size), y_true] + self.eps)
        )
        return loss

    def backward(self, logits, y_true):
        batch_size = y_true.shape[0]

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        grad = probs
        grad[np.arange(batch_size), y_true] -= 1
        grad /= batch_size
        return grad
