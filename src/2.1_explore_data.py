import wandb
import numpy as np
from utils.data_loader import load_dataset

mnist_classes = {i: str(i) for i in range(10)}

fashion_classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def log_dataset(name, class_map):
    X_train, y_train, _, _ = load_dataset(name)

    table = wandb.Table(
        columns=["class_id", "class_name",
                 "img1", "img2", "img3", "img4", "img5"]
    )

    for cls in range(10):
        idxs = np.where(y_train == cls)[0][:5]

        images = [
            wandb.Image(X_train[idx].reshape(28, 28))
            for idx in idxs
        ]

        table.add_data(
            cls,
            class_map[cls],
            images[0], images[1], images[2],
            images[3], images[4]
        )

    wandb.log({f"{name}_samples": table})


def main():
    wandb.init(
        project="da6401-assignment1",
        name="data-exploration",
        tags=["task-2.1", "data-exploration"]
    )

    log_dataset("mnist", mnist_classes)
    log_dataset("fashion_mnist", fashion_classes)

    wandb.finish()


if __name__ == "__main__":
    main()