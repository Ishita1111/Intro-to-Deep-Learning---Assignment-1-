"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np
import wandb
import copy
import sys

from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MLP using NumPy only")

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["mean_squared_error", "cross_entropy"])

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-lr", "--lr", "--learning_rate",dest="lr",type=float,required=True,
                        help="Learning rate")
    parser.add_argument("-wd", "--wd", dest="wd", type=float, default=0.0,
                        help="Weight decay (L2 regularisation)")

    parser.add_argument("-nhl", "--nhl", "--num_layers", dest="num_layers", type=int)
    parser.add_argument("-sz", "--sz", "--hidden_size", dest="hidden_size", nargs="+", type=int)
    parser.add_argument("-w_i", "--w_i", "--weight_init", dest="weight_init", choices=["random","xavier","zeroes"])
    
    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function")

    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401-assignment1")

    return parser.parse_args()


def one_hot_encode(y, num_classes=10):
    y_encoded = np.zeros((y.shape[0], num_classes))
    y_encoded[np.arange(y.shape[0]), y] = 1
    return y_encoded

def compute_f1_score(y_true, y_pred, num_classes=10):
    f1_scores = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return np.mean(f1_scores)


def main():

    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # models_dir = os.path.join(project_root, "models")
    models_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(models_dir, exist_ok=True)

    # If no CLI args provided, load best saved config
    if len(sys.argv) == 1:
        config_path = os.path.join(models_dir, "best_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            args = argparse.Namespace(**saved_config)
            args.wandb_project = "da6401-assignment1"
            print("Loaded best config from models folder.")
        else:
            raise ValueError("No saved best_config.json found.")
    else:
        args = parse_arguments()
    
    if args.hidden_size is None:
        args.hidden_size = [64] * args.num_layers

    wandb.init(
        project=args.wandb_project, # wandb project name : da6401-assignment1
        config=vars(args),
        group="Tests-during-grading",
        name=f"{args.dataset}-{args.optimizer}-{args.num_layers}-{args.activation}-{args.weight_init}"
    )
    config = wandb.config

    # load data 
    X_train, y_train, X_test, y_test = load_dataset(config.dataset)

    if config.loss == "mean_squared_error":
        y_train = one_hot_encode(y_train)
        y_test = one_hot_encode(y_test) 

    # initialize model
    model = NeuralNetwork(config)

    best_f1 = -1.0
    best_state = None

    for epoch in range(config.epochs):
        loss, grad_norm = model.train(X_train, y_train, epochs=1, batch_size=config.batch_size, log_gradients=True)

        # evaluate after each epoch
        train_acc = model.evaluate(X_train, y_train)
        test_acc = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)

        if config.loss == "mean_squared_error":
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test

        test_f1 = compute_f1_score(y_true, y_pred)
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_f1":test_f1
        }

        # Logging for 2.5 - Dead Neurons
        # dead_fractions, mean_activations = model.compute_dead_neurons(X_test)
        # for i, hidden in enumerate(model.hidden_outputs):
        #     log_dict[f"layer{i+1}_activation_hist"] = wandb.Histogram(hidden)   
        # for i, dead in enumerate(dead_fractions):
        #     log_dict[f"dead_fraction_layer{i+1}"] = dead
        # for i, act in enumerate(mean_activations):
        #     log_dict[f"mean_activation_layer{i+1}"] = act

        # grad_norms = [
        #     np.linalg.norm(layer.grad_W)
        #     for layer in model.layers[:-1]  # exclude output layer
        # ]

        # for i, g in enumerate(grad_norms):
        #     log_dict[f"grad_norm_layer{i+1}"] = g
        
        wandb.log(log_dict)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = copy.deepcopy(model)

        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {loss:.4f} -Train Accuracy: {train_acc:.4f} - Test Accuracy: {test_acc:.4f}")

    if best_state is None:
        print("Warning: no epochs were run — nothing to save.")
        return

    # ---------------------------------
    # SAFE SAVE (Prevents Overwrite)
    # ---------------------------------

    best_config_path = os.path.join(models_dir, "best_config.json")
    weights_path = os.path.join(models_dir, "best_model.npy")

    save_model = True

    if os.path.exists(best_config_path):
        with open(best_config_path, "r") as f:
            existing_config = json.load(f)

        if "best_score" in existing_config:
            if existing_config["best_score"] >= best_f1:
                save_model = False
                print("Existing best model has equal or better F1 score. Not overwriting.")

    if save_model:
        best_config = dict(config)
        best_config["best_score"] = best_f1

        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=4)

        weights = [
            {"W": layer.W, "b": layer.b}
            for layer in best_state.layers
        ]

        np.save(weights_path, {"weights": weights}, allow_pickle=True)

        print("Saved new best model based on F1 score.")
    else:
        print("No new best model saved.")

    artifact = wandb.Artifact("best-model", type="model")
    artifact.add_file(weights_path)
    artifact.add_file(best_config_path)
    wandb.log_artifact(artifact)

    print("Training complete!")
    print(f"Best Test F1 Score: {best_f1:.4f}")

    # =============================
    # 2.8 CONFUSION MATRIX EVALUATION
    # =============================

    """
    saved = np.load(os.path.join(models_dir, "best_model.npy"), allow_pickle=True).item()
    saved_weights = saved["weights"]

    for layer, w in zip(model.layers, saved_weights):
        layer.W = w["W"]
        layer.b = w["b"]

    # Predictions
    y_pred = model.predict(X_test)

    if config.loss == "mean_squared_error":
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Log standard confusion matrix
    wandb.log({
        "confusion_matrix":
            wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
            preds=y_pred,
            class_names=[str(i) for i in range(10)]
        )
    })
    

    # =============================
    # 2.9 CREATIVE FAILURE VISUALIZATION
    # =============================

    # Show first 50 incorrect predictions
    incorrect_indices = np.where(y_true != y_pred)[0][:50]

    wandb.log({
        "incorrect_examples":
            [wandb.Image(X_test[i].reshape(28, 28))
            for i in incorrect_indices]
    })
    """
    wandb.finish()  


if __name__ == "__main__":
    main()

