# Evaluation Metrics

This module computes classification metrics and saves visualizations for the
Cross-Out Detection project.

## Metrics

- Overall top-1 accuracy
- Per-class precision, recall, F1, and support
- Macro precision, recall, and F1
- Optional CLEAN vs non-CLEAN binary accuracy when `CLEAN` is present

## CLI Usage

Multiclass checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/multiclass/best.pth \
  --data-dir data/ImageFolder
```

Binary checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/binary/best.pth \
  --data-dir data/ImageFolder \
  --binary
```

Outputs are written to `evaluation_outputs/` by default:

```text
metrics.json
confusion_matrix.png
training_curves.png
```
