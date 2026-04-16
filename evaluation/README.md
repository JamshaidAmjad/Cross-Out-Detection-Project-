# Evaluation Metrics

This module computes classification metrics and produces visualisations for the Cross-Out Detection project.

## What to Implement

### `metrics.py`

| Function | Description |
|----------|-------------|
| `compute_accuracy(preds, labels)` | Overall top-1 accuracy |
| `compute_per_class_metrics(preds, labels, class_names)` | Precision, recall, F1 per class |
| `compute_macro_metrics(preds, labels)` | Macro-averaged precision, recall, F1 |
| `compute_binary_accuracy(preds, labels)` | Accuracy treating CLEAN (0) vs all others |

All functions should accept either torch `Tensor` or numpy array inputs.
Use `sklearn.metrics` where appropriate.

### `visualize.py`

| Function | Description |
|----------|-------------|
| `plot_confusion_matrix(preds, labels, class_names, save_path)` | Seaborn heatmap of the normalised confusion matrix |
| `plot_training_curves(train_losses, val_losses, val_accuracies, save_path)` | Side-by-side loss and accuracy curves |

Both functions should save the figure to `save_path` and optionally display it.

## Usage

```python
from evaluation.metrics import compute_accuracy, compute_macro_metrics
from evaluation.visualize import plot_confusion_matrix

acc = compute_accuracy(preds, labels)
metrics = compute_macro_metrics(preds, labels)
plot_confusion_matrix(preds, labels, class_names, save_path="outputs/cm.png")
```
