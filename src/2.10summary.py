import wandb

api = wandb.Api()

# Replace with your actual entity/project
runs = api.runs("me22b134-indian-institute-of-technology-madras/da6401-assignment1")

wandb.init(
    project="da6401-assignment1",
    group="2.10-Fashion-MNIST",
    name="fashion-mnist-summary"
)

table = wandb.Table(columns=[
    "Run Name",
    "Layers",
    "Hidden Size",
    "Optimizer",
    "Activation",
    "Train Accuracy",
    "Test Accuracy"
])

for run in runs:
    config = run.config

    # Only include Fashion-MNIST runs
    if config.get("dataset") == "fashion_mnist":

        summary = run.summary

        table.add_data(
            run.name,
            config.get("num_layers"),
            config.get("hidden_size"),
            config.get("optimizer"),
            config.get("activation"),
            summary.get("train_accuracy"),
            summary.get("test_accuracy")
        )

wandb.log({"Fashion-MNIST Comparison Table": table})
wandb.finish()