"""
Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np

from .neural_layer import NeuralLayer
from .activations import Sigmoid, Tanh, ReLU
from .objective_functions import MeanSquaredError, CrossEntropy
from .optimizers import Optimizer
# import wandb


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        self.activations = []

        input_dim = 784  # MNIST / Fashion-MNIST flattened
        hidden_sizes = cli_args.hidden_size
        num_layers = cli_args.num_layers

        # activation choice
        if cli_args.activation == "sigmoid":
            act_class = Sigmoid
        elif cli_args.activation == "tanh":
            act_class = Tanh
        else:
            act_class = ReLU

        # build hidden layers
        prev_dim = input_dim
        for i in range(num_layers):
            layer = NeuralLayer(
                prev_dim,
                hidden_sizes[i],
                weight_init=cli_args.weight_init
            )
            self.layers.append(layer)
            self.activations.append(act_class())
            prev_dim = hidden_sizes[i]

        # output layer (10 classes)
        self.layers.append(
            NeuralLayer(prev_dim, 10, weight_init=cli_args.weight_init)
        )

        # loss
        if cli_args.loss == "mean_squared_error":
            self.loss_fn = MeanSquaredError()
        else:
            self.loss_fn = CrossEntropy()

        # optimizer (single object handling all layers)
        self.optimizer = Optimizer(
            lr=cli_args.lr,
            weight_decay=getattr(cli_args, 'wd', 0.0),
            optimizer_type=cli_args.optimizer
        )

        # Will be populated after each backward() call:
        # list of (grad_W, grad_b) tuples, one per layer, in forward order.
        self.grads = []

    def forward(self, X, store_activations=False):
        """
        Forward propagation through all layers.
        """
        out = X
        self.hidden_outputs = []  # store hidden layer outputs
        for i in range(len(self.activations)):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)
            if store_activations:
                self.hidden_outputs.append(out)
        out = self.layers[-1].forward(out)
        return out

    def backward(self, y_true, logits):
        """
        Backward propagation to compute gradients.

        Args:
            y_true: Ground truth labels
            logits: Raw outputs from the network (before softmax)

        After this call every NeuralLayer exposes:
            layer.grad_W - gradient w.r.t. weights
            layer.grad_b - gradient w.r.t. biases
        """
        # Gradient of loss w.r.t logits
        grad = self.loss_fn.backward(logits, y_true)
        # Backprop through output layer
        grad = self.layers[-1].backward(grad)
        # Backprop through hidden layers (reverse order)
        for i in reversed(range(len(self.activations))):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

        # Collect gradients for verification
        self.grads = [(layer.grad_W, layer.grad_b) for layer in self.layers]

        return grad

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.step(self.layers)

    def get_weights(self):
        """
        Return all layer weights and biases in a serializable format.
        """
        weights = []
        for layer in self.layers:
            weights.append({
                "W": layer.W,
                "b": layer.b
            })
        return weights

    def set_weights(self, weights):
        """
        Load weights into the network layers.
        """
        for layer, w in zip(self.layers, weights):
            layer.W = w["W"]
            layer.b = w["b"]

    def train(self, X_train, y_train, epochs, batch_size, log_gradients=False):

        n = X_train.shape[0]
        global_step=0
        for epoch in range(epochs):

            perm = np.random.permutation(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            num_batches = 0

            for i in range(0, n, batch_size):

                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(logits, y_batch)

                epoch_loss += loss
                
                self.backward(y_batch, logits)

                if log_gradients and global_step < 50:
                    layer = self.layers[0]  # first hidden layer
                    num_neurons = layer.grad_W.shape[1]
                    neuron_ids = list(range(min(5, num_neurons)))

                    log_dict = {"iteration": global_step}

                    for nid in neuron_ids:
                        grad_vector = layer.grad_W[:, nid]
                        log_dict[f"neuron_{nid}_grad"] = np.linalg.norm(grad_vector)    
                    
#                    wandb.log(log_dict)

                    global_step += 1

                # Compute gradient norm of first hidden layer
                grad_norm = np.linalg.norm(self.layers[0].grad_W)
                epoch_grad_norm += grad_norm

                self.update_weights()

                num_batches += 1

            epoch_loss /= num_batches
            epoch_grad_norm /= num_batches

            return epoch_loss, epoch_grad_norm

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)

        # If y is one-hot (MSE case)
        if y.ndim == 2:
            labels = np.argmax(y, axis=1)
        else:
            labels = y

        accuracy = np.mean(predictions == labels)
        return accuracy

    def predict(self, X):
        """
        Predict class labels for input data.
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def compute_dead_neurons(self, X):

        # Forward pass storing activations
        _ = self.forward(X, store_activations=True)
        dead_fractions = []
        mean_activations = []
        
        for hidden in self.hidden_outputs:
            # hidden shape: (N, neurons)
            dead = np.mean(hidden <= 1e-6, axis=0)
            dead_fraction = np.mean(dead > 0.95)
            dead_fractions.append(dead_fraction)
            mean_activation = np.mean(np.abs(hidden), axis=0)
            mean_activations.append(mean_activation)
        return dead_fractions, mean_activations 