# Multi-Layer Perceptron (NumPy Implementation)

This project implements a fully connected neural network (MLP) from
scratch using only NumPy. The goal of this assignment was to understand
every component of neural network training manually - including
forward propagation, backpropagation, optimization, and evaluation -
without using deep learning frameworks.

The model supports training on MNIST and Fashion-MNIST datasets and
includes multiple optimizers, activation functions, and weight
initialization strategies.

------------------------------------------------------------------------

## Project Structure

    A1/
    ├── models/
    │   ├── best_config.json
    │   └── best_model.npy
    │
    ├── src/
    │   ├── ann/
    │   │   ├── __init__.py
    │   │   ├── activations.py
    │   │   ├── neural_layer.py
    │   │   ├── neural_network.py
    │   │   ├── objective_functions.py
    │   │   └── optimizers.py
    │   │
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   └── data_loader.py
    │   │
    │   ├── train.py
    │   └── inference.py
    │
    └── README.md
    └── requirements.txt

------------------------------------------------------------------------

## Features

### Activation Functions

-   Sigmoid\
-   Tanh\
-   ReLU

### Loss Functions

-   Mean Squared Error\
-   Cross Entropy (with numerically stable softmax)

### Optimizers

-   SGD\
-   Momentum\
-   NAG\
-   RMSProp\
-   Adam\
-   Nadam

### Weight Initialization

-   Random (small Gaussian initialization)\
-   Xavier initialization\
-   Zero initialization (for experimentation)

------------------------------------------------------------------------

## Training the Model

Example:

``` bash
python src/train.py -d mnist                     -e 10                     -b 64                     -l cross_entropy                     -o adam                     --lr 0.001                     --nhl 2                     --sz 128 64                     -a relu                     --w_i xavier
```

This will: - Train the model - Log metrics - Save the best performing
model inside the `models/` directory

Saved files: - `best_model.npy` - `best_config.json`

------------------------------------------------------------------------

## Running Inference

To evaluate the best saved model:

``` bash
python src/inference.py
```

You can also manually specify weights and configuration if needed.

The script prints: - Loss - Accuracy - Precision - Recall - F1-score

------------------------------------------------------------------------

## Implementation Details

Each fully connected layer computes:

z = XW + b

Hidden layers apply a chosen activation function.

For Cross-Entropy loss with softmax:

dL/dz = (1/N)(p - y)

Gradients are computed analytically using vectorized NumPy operations.

------------------------------------------------------------------------

## Requirements

-   Python 3.x
-   NumPy
-   scikit-learn
-   Keras (for dataset loading)
-   Weights & Biases (if logging is enabled)

------------------------------------------------------------------------

