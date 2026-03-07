import numpy as np


class Optimizer:
    def __init__(self, lr=0.001, weight_decay=0.0, optimizer_type="sgd"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type

        self.velocity_W = {}
        self.velocity_b = {}

        self.cache_W = {}
        self.cache_b = {}

        self.m_W = {}
        self.m_b = {}
        self.v_W = {}
        self.v_b = {}

        self.t = 0

    def step(self, layers):
        if self.optimizer_type == "sgd":
            self._sgd(layers)
        elif self.optimizer_type == "momentum":
            self._momentum(layers)
        elif self.optimizer_type == "nag":
            self._nag(layers)
        elif self.optimizer_type == "rmsprop":
            self._rmsprop(layers)
        elif self.optimizer_type == "adam":
            self._adam(layers)
        elif self.optimizer_type == "nadam":
            self._nadam(layers)
        else:
            raise ValueError("Unknown optimizer")

    def _sgd(self, layers):
        for idx, layer in enumerate(layers):
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b

    def _momentum(self, layers, beta=0.9):
        for idx, layer in enumerate(layers):
            if idx not in self.velocity_W:
                self.velocity_W[idx] = np.zeros_like(layer.W)
                self.velocity_b[idx] = np.zeros_like(layer.b)

            self.velocity_W[idx] = beta * self.velocity_W[idx] + layer.grad_W
            self.velocity_b[idx] = beta * self.velocity_b[idx] + layer.grad_b

            layer.W -= self.lr * (self.velocity_W[idx] + self.weight_decay * layer.W)
            layer.b -= self.lr * self.velocity_b[idx]

    def _nag(self, layers, beta=0.9):
        for idx, layer in enumerate(layers):
            if idx not in self.velocity_W:
                self.velocity_W[idx] = np.zeros_like(layer.W)
                self.velocity_b[idx] = np.zeros_like(layer.b)

            v_prev_W = self.velocity_W[idx]
            v_prev_b = self.velocity_b[idx]

            self.velocity_W[idx] = beta * self.velocity_W[idx] + layer.grad_W
            self.velocity_b[idx] = beta * self.velocity_b[idx] + layer.grad_b

            layer.W -= self.lr * (beta * v_prev_W + self.velocity_W[idx] + self.weight_decay * layer.W)
            layer.b -= self.lr * (beta * v_prev_b + self.velocity_b[idx])

    def _rmsprop(self, layers, beta=0.9, eps=1e-8):
        for idx, layer in enumerate(layers):
            if idx not in self.cache_W:
                self.cache_W[idx] = np.zeros_like(layer.W)
                self.cache_b[idx] = np.zeros_like(layer.b)

            self.cache_W[idx] = beta * self.cache_W[idx] + (1 - beta) * (layer.grad_W ** 2)
            self.cache_b[idx] = beta * self.cache_b[idx] + (1 - beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.cache_W[idx]) + eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.cache_b[idx]) + eps)

    def _adam(self, layers, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        for idx, layer in enumerate(layers):
            if idx not in self.m_W:
                self.m_W[idx] = np.zeros_like(layer.W)
                self.v_W[idx] = np.zeros_like(layer.W)
                self.m_b[idx] = np.zeros_like(layer.b)
                self.v_b[idx] = np.zeros_like(layer.b)

            self.m_W[idx] = beta1 * self.m_W[idx] + (1 - beta1) * layer.grad_W
            self.v_W[idx] = beta2 * self.v_W[idx] + (1 - beta2) * (layer.grad_W ** 2)

            self.m_b[idx] = beta1 * self.m_b[idx] + (1 - beta1) * layer.grad_b
            self.v_b[idx] = beta2 * self.v_b[idx] + (1 - beta2) * (layer.grad_b ** 2)

            m_hat_W = self.m_W[idx] / (1 - beta1 ** self.t)
            v_hat_W = self.v_W[idx] / (1 - beta2 ** self.t)

            m_hat_b = self.m_b[idx] / (1 - beta1 ** self.t)
            v_hat_b = self.v_b[idx] / (1 - beta2 ** self.t)

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def _nadam(self, layers, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        for idx, layer in enumerate(layers):
            if idx not in self.m_W:
                self.m_W[idx] = np.zeros_like(layer.W)
                self.v_W[idx] = np.zeros_like(layer.W)
                self.m_b[idx] = np.zeros_like(layer.b)
                self.v_b[idx] = np.zeros_like(layer.b)

            self.m_W[idx] = beta1 * self.m_W[idx] + (1 - beta1) * layer.grad_W
            self.v_W[idx] = beta2 * self.v_W[idx] + (1 - beta2) * (layer.grad_W ** 2)

            self.m_b[idx] = beta1 * self.m_b[idx] + (1 - beta1) * layer.grad_b
            self.v_b[idx] = beta2 * self.v_b[idx] + (1 - beta2) * (layer.grad_b ** 2)

            m_hat_W = self.m_W[idx] / (1 - beta1 ** self.t)
            v_hat_W = self.v_W[idx] / (1 - beta2 ** self.t)

            m_hat_b = self.m_b[idx] / (1 - beta1 ** self.t)
            v_hat_b = self.v_b[idx] / (1 - beta2 ** self.t)

            nesterov_W = beta1 * m_hat_W + (1 - beta1) * layer.grad_W / (1 - beta1 ** self.t)
            nesterov_b = beta1 * m_hat_b + (1 - beta1) * layer.grad_b / (1 - beta1 ** self.t)

            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + eps)
