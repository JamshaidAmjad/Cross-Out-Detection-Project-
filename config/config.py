"""
config.py
---------
Centralised hyperparameter and path configuration for the Cross-Out
Detection project.

All training scripts import their settings from this module so that
experiments can be reproduced by changing values in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path

# TODO: import torch (needed for device auto-detection)


@dataclass
class Config:
    """Experiment configuration.

    Attributes:
        data_dir: Root directory containing per-class image subfolders.
        checkpoint_dir: Directory to save model checkpoints.
        num_classes: Number of output classes (8 for this project).
        image_size: Input image resolution required by ViT (224).
        batch_size: Number of samples per training mini-batch.
        num_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for AdamW optimiser.
        weight_decay: L2 regularisation coefficient for AdamW.
        label_smoothing: Label smoothing factor for CrossEntropyLoss.
        early_stopping_patience: Epochs to wait before early stopping.
        device: Training device ('cuda' or 'cpu').
        num_workers: DataLoader worker processes.
        seed: Global random seed for reproducibility.
    """

    data_dir: str = "data/dataset"
    checkpoint_dir: str = "checkpoints/"
    num_classes: int = 8
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    num_workers: int = 4
    seed: int = 42

    # TODO: Set device automatically:
    #       device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device: str = "cpu"  # replace with auto-detection above
