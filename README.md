# Cross-Out Detection Project

A Vision Transformer (ViT) based system for classifying cross-out annotations in handwritten or scanned documents into 8 categories.

## Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | CLEAN | No cross-out present |
| 1 | CROSS | X-shaped cross-out |
| 2 | DIAGONAL | Single diagonal line |
| 3 | DOUBLE_LINE | Two parallel lines |
| 4 | SCRATCH | Irregular scribble |
| 5 | SINGLE_LINE | One horizontal line |
| 6 | WAVE | Wavy/curved line |
| 7 | ZIG_ZAG | Zig-zag pattern |

> The `MIXED` folder in the dataset is excluded from training.

---

## Project Structure

```
Cross-Out-Detection-Project/
├── data/                   # Dataset loader and augmentations
│   ├── dataset.py          # ImageFolder loader + get_dataloaders()
│   └── transforms.py       # Train/val transform pipelines
├── models/                 # Model architecture
│   └── vit_model.py        # ViT-Base/16 via timm + get_model()
├── config/                 # Hyperparameters
│   └── config.py           # Config dataclass
├── training/               # Training loop
│   └── trainer.py          # Trainer class
├── evaluation/             # Metrics and visualisations
│   ├── metrics.py          # Accuracy, F1, binary accuracy
│   └── visualize.py        # Confusion matrix, training curves
├── custom_dataset/         # Manually created images (~50 samples)
├── scripts/
│   ├── train.py            # Training entry point
│   ├── evaluate.py         # Evaluation entry point
│   └── inference.py        # Single-image prediction
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/JamshaidAmjad/Cross-Out-Detection-Project-.git
cd Cross-Out-Detection-Project-

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training

```bash
python scripts/train.py
```

Edit `config/config.py` to change hyperparameters (learning rate, batch size, epochs, etc.) or point `data_dir` at your dataset.

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pth
```

Outputs per-class precision/recall/F1, macro averages, binary CLEAN vs. non-CLEAN accuracy, and saves a confusion matrix heatmap.

---

## Inference

```bash
python scripts/inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best.pth \
    --top-k 3
```

---

## Requirements

See `requirements.txt`. Key dependencies: PyTorch, timm, torchvision, scikit-learn, seaborn, matplotlib.
