"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os, sys 
import json

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork


def parse_arguments():

    parser = argparse.ArgumentParser(description="Inference using trained MLP")

    # Dataset
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on")

    # Training hyperparameters (must match train.py)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of epochs")

    parser.add_argument("-b", "--batch_size", type=int,
                        help="Mini-batch size")

    parser.add_argument("-l", "--loss", type=str,
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function")

    parser.add_argument("-o", "--optimizer", type=str,
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer")

    parser.add_argument("-lr", "--lr", "--learning_rate", type=float,
                        help="Learning rate")

    parser.add_argument("-wd", "--wd", type=float, default=0.0,
                        help="Weight decay (L2 regularisation)")

    # Architecture
    parser.add_argument("-nhl", "--nhl", "--num_layers", dest="num_layers",
                        type=int,
                        help="Number of hidden layers")

    parser.add_argument("-sz", "--sz", "--hidden_size", dest="hidden_size",
                        nargs="+", type=int,
                        help="Hidden layer size(s)")

    parser.add_argument("-a", "--activation", type=str,
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function")

    parser.add_argument("-w_i", "--w_i", "--weight_init", dest="weight_init",
                        choices=["random", "xavier", "zeroes"],
                        help="Weight initialisation method")

    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401-assignment1")

    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data

def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot vectors."""
    y_enc = np.zeros((y.shape[0], num_classes))
    y_enc[np.arange(y.shape[0]), y] = 1
    return y_enc


def evaluate_model(model, X_test, y_test_onehot):
    """
    Run forward pass and compute Accuracy, Precision, Recall, and F1-score.

    Args:
        model         : NeuralNetwork instance
        X_test        : ndarray, shape (N, 784)
        y_test_onehot : ndarray, shape (N, 10) one-hot encoded labels

    Returns:
        dict with keys: logits, loss, accuracy, precision, recall, f1
    """
    logits = model.forward(X_test)

    y_pred = np.argmax(logits, axis=1)

    if len(y_test_onehot.shape) == 2:
        y_true = np.argmax(y_test_onehot, axis=1)
    else:
        y_true = y_test_onehot  

    loss = model.loss_fn.forward(logits, y_test_onehot)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "logits"   : logits,
        "loss"     : float(loss),
        "accuracy" : float(accuracy),
        "precision": float(precision),
        "recall"   : float(recall),
        "f1"       : float(f1),
    }


def main():
    """
    Main inference function.
    Loads serialized NumPy weights, runs evaluation, and prints metrics.
    """
    args = parse_arguments()

    models_dir = os.path.dirname(__file__)
    config_path = os.path.join(models_dir, "best_config.json")
    weights_path = os.path.join(models_dir, "best_model.npy")

    # Fill missing args from best_config
    if os.path.exists(config_path):

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        for key, value in saved_config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

        print("Filled missing arguments using best_config.json")

    # DEBUG (optional)
    print("ARGS AFTER FILL:", args)

    # Load test split
    _, _, X_test, y_test_raw = load_dataset(args.dataset)

    if args.loss == "mean_squared_error":
        y_test = one_hot_encode(y_test_raw)
    else:
        y_test = y_test_raw 

    # Reconstruct model and inject saved weights
    model = NeuralNetwork(args)
    weights = load_model(weights_path)
    if isinstance(weights, dict) and "weights" in weights:
        weights = weights["weights"]
    model.set_weights(weights)
    results = evaluate_model(model, X_test, y_test)

    y_pred = np.argmax(results["logits"], axis=1)

    if len(y_test.shape) == 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test 

    print("\n" + "=" * 40)
    print(f"  Dataset   : {args.dataset}")
    print("=" * 40)
    print(f"  Loss      : {results['loss']:.6f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-score  : {results['f1']:.4f}")
    print("=" * 40 + "\n")

    
    wandb.init(
        project=args.wandb_project, # wandb project name : da6401-assignment1
        config=vars(args),
        group="Tests-during-grading",
        name=f"{args.dataset}-{args.optimizer}-{args.num_layers}-{args.activation}-{args.weight_init}"
    )

    wandb.log({
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
        "test_precision": results["precision"],
        "test_recall": results["recall"],
        "test_f1": results["f1"]
    })
    
    wandb.finish()

    """
    # CONFUSION MATRIX


    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues")

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=15)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()


    plt.close(fig)
    """

    return results


if __name__ == '__main__':
    main()
