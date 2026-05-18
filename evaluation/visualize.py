"""
visualize.py
------------
Visualisation utilities for evaluation outputs.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

ArrayLike = Union[np.ndarray, torch.Tensor, List[int]]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def plot_confusion_matrix(
    preds: ArrayLike,
    labels: ArrayLike,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot and optionally save a normalized confusion matrix heatmap."""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    class_indices = list(range(len(class_names)))
    matrix = confusion_matrix(
        labels_np,
        preds_np,
        labels=class_indices,
        normalize="true",
    )
    matrix = np.nan_to_num(matrix)

    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot training/validation loss and validation accuracy curves."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="Train loss")
    axes[0].plot(epochs, val_losses, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_accuracies, label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
