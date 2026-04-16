"""
visualize.py
------------
Visualisation utilities for the Cross-Out Detection project.

Provides:
  - plot_confusion_matrix: normalised heatmap using seaborn
  - plot_training_curves: side-by-side loss and accuracy plots
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# TODO: import matplotlib.pyplot as plt
# TODO: import seaborn as sns
# TODO: from sklearn.metrics import confusion_matrix

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


def plot_confusion_matrix(
    preds: ArrayLike,
    labels: ArrayLike,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot and optionally save a normalised confusion matrix heatmap.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).
        class_names: List of class name strings for axis labels.
        save_path: If provided, save the figure to this path.
        show: If True, call plt.show() to display the figure.
    """
    # TODO: Convert tensors to numpy if necessary
    # TODO: Compute confusion_matrix(labels, preds, normalize='true')
    # TODO: Create a plt.figure with appropriate size
    # TODO: Use sns.heatmap with annot=True, fmt='.2f', xticklabels and
    #       yticklabels set to class_names
    # TODO: Set axis labels ('Predicted', 'True') and title
    # TODO: Save figure to save_path if provided (plt.savefig)
    # TODO: Call plt.show() if show=True
    # TODO: Close the figure to free memory (plt.close)
    raise NotImplementedError("Implement plot_confusion_matrix()")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot training/validation loss and validation accuracy curves.

    Args:
        train_losses: List of training loss values (one per epoch).
        val_losses: List of validation loss values (one per epoch).
        val_accuracies: List of validation accuracy values (one per epoch).
        save_path: If provided, save the figure to this path.
        show: If True, call plt.show() to display the figure.
    """
    # TODO: Create a figure with two side-by-side subplots
    # TODO: Left subplot: plot train_losses and val_losses vs epoch index
    #       with legend, xlabel='Epoch', ylabel='Loss'
    # TODO: Right subplot: plot val_accuracies vs epoch index
    #       xlabel='Epoch', ylabel='Accuracy'
    # TODO: Add an overall title and tight_layout()
    # TODO: Save and/or show as above
    # TODO: Close the figure
    raise NotImplementedError("Implement plot_training_curves()")
