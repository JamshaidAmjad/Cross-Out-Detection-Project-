"""
metrics.py
----------
Classification metrics for the Cross-Out Detection project.
"""

from typing import Dict, List, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ArrayLike = Union[np.ndarray, torch.Tensor, List[int]]

CLEAN_CLASS_IDX = 0


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def compute_accuracy(preds: ArrayLike, labels: ArrayLike) -> float:
    """Compute overall top-1 accuracy."""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    return float(accuracy_score(labels_np, preds_np))


def compute_per_class_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, and F1 for each class."""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    class_indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np,
        preds_np,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    return {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }


def compute_macro_metrics(
    preds: ArrayLike,
    labels: ArrayLike,
) -> Dict[str, float]:
    """Compute macro-averaged precision, recall, and F1 across all classes."""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        average="macro",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_binary_accuracy(
    preds: ArrayLike,
    labels: ArrayLike,
    clean_class_idx: int = CLEAN_CLASS_IDX,
) -> float:
    """Compute CLEAN vs. non-CLEAN accuracy."""
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    binary_preds = (preds_np != clean_class_idx).astype(int)
    binary_labels = (labels_np != clean_class_idx).astype(int)
    return float(accuracy_score(binary_labels, binary_preds))
