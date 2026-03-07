"""
Data loading utilities for MNIST and Fashion-MNIST datasets.
Handles downloading (via Keras), normalisation, and flattening.
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist


def load_dataset(name: str):
    """
    Load, normalise, and flatten a dataset.

    Args:
        name: One of "mnist" or "fashion_mnist".

    Returns:
        Tuple (X_train, y_train, X_test, y_test) where:
            X_*  : float32 ndarray, shape (N, 784), values in [0, 1]
            y_*  : int ndarray, shape (N,), class indices 0-9
    """
    
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'.")

    # Normalise pixel values to [0, 1]
    x_train = x_train.astype(np.float64) / 255.0
    x_test  = x_test.astype(np.float64)  / 255.0

    # Flatten 28x28 images to 784-dimensional vectors
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0],  -1)

    return x_train, y_train, x_test, y_test
