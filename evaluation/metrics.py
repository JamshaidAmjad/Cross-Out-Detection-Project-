"""
metrics.py
----------
Classification metrics for the Cross-Out Detection project.

Provides per-class and macro-averaged precision, recall, and F1-score,
overall accuracy, and a binary CLEAN vs. non-CLEAN accuracy metric.

All functions accept either torch.Tensor or numpy.ndarray inputs and
convert internally as needed.
"""

from typing import Dict, List, Union

import numpy as np

# TODO: from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support, classification_report
# )
# TODO: import torch

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821

CLEAN_CLASS_IDX = 0


def compute_accuracy(preds: ArrayLike, labels: ArrayLike) -> float:
    """Compute overall top-1 accuracy.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    # TODO: Convert tensors to numpy if necessary
    # TODO: Use sklearn.metrics.accuracy_score(labels, preds)
    raise NotImplementedError("Implement compute_accuracy()")


def compute_per_class_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, and F1 for each class.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).
        class_names: List of class name strings (length == num_classes).

    Returns:
        Dict mapping class_name -> {'precision': ..., 'recall': ..., 'f1': ...}
    """
    # TODO: Use precision_recall_fscore_support(..., average=None)
    # TODO: Build and return the dict keyed by class_names
    raise NotImplementedError("Implement compute_per_class_metrics()")


def compute_macro_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
) -> Dict[str, float]:
    """Compute macro-averaged precision, recall, and F1 across all classes.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    # TODO: Use precision_recall_fscore_support(..., average='macro')
    # TODO: Return {'precision': ..., 'recall': ..., 'f1': ...}
    raise NotImplementedError("Implement compute_macro_metrics()")


def compute_binary_accuracy(
    preds: ArrayLike,
    labels: ArrayLike,
) -> float:
    """Compute binary accuracy: CLEAN (class 0) vs. all other classes.

    Args:
        preds: Predicted class indices, shape (N,).
        labels: Ground-truth class indices, shape (N,).

    Returns:
        Binary accuracy as a float in [0, 1].
    """
    # TODO: Convert preds and labels to binary: 0 if == CLEAN_CLASS_IDX, else 1
    # TODO: Return accuracy_score on the binary arrays
    raise NotImplementedError("Implement compute_binary_accuracy()")
