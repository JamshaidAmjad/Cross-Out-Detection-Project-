import wandb

import json
import os

checkpoints = "checkpoints"

models = [f for f in os.listdir(checkpoints) if os.path.isdir(os.path.join(checkpoints, f))]

def handle_model(model):
    path = os.path.join(checkpoints, model)
    submodels = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    for sub in submodels:
        metadata_path = os.path.join(path, sub, "metadata.json")
        with open(metadata_path, "r") as f:
            data = json.load(f)
        upload(model, f"{model}-{sub}", data)


def upload(model, name, metadata):
    config = metadata['cfg']

    num_actual_epochs = len(metadata["train_losses"])

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sammel-2-lule-university-of-technology",
        # Set the wandb project where this run will be logged.
        project="document-analysis",

        name=name,

        # Track hyperparameters and run metadata.
        config={
            "dataset": "IAM Cross-out",
            "architecture": f"{model}",
            "num_classes": config["num_classes"],

            "epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "label_smoothing": config["label_smoothing"],
            "early_stopping_patience": config["early_stopping_patience"],
            "actual_epochs": num_actual_epochs,

            "training_time": metadata["training_time"],

        },
    )

    for i in range(num_actual_epochs):
        run.log({
            "train_loss": metadata["train_losses"][i],
            "val_loss": metadata["val_losses"][i],
            "train_accuracy": metadata["train_accuracies"][i],
            "val_accuracy": metadata["val_accuracies"][i],
        })
    

    run.finish()



for i in models:
    handle_model(i)
