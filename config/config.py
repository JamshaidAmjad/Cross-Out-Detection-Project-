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
import torch

# TODO: import torch (needed for device auto-detection)


@dataclass
class Config:
    """Experiment configuration.

    Attributes:
        data_dir: Root directory containing per-class image subfolders.
        checkpoint_dir: Directory to save model checkpoints.
        num_classes: Number of output classes for the selected mode.
        image_size: Input image resolution required by ViT (224).
        batch_size: Number of samples per training mini-batch.
        num_epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for AdamW optimiser.
        weight_decay: L2 regularisation coefficient for AdamW.
        label_smoothing: Label smoothing factor for CrossEntropyLoss.
        early_stopping_patience: Epochs to wait before early stopping.
        device: Training device ('cuda' or 'cpu'), auto-detected.
        num_workers: DataLoader worker processes.
        seed: Global random seed for reproducibility.
        multiclass: If True, train on all 7 cross-out types.
                    If False, train binary CLEAN vs. crossed-out (2-class).
    """

    data_dir: str = "data/ImageFolder/"
    checkpoint_dir: str = "checkpoints/"
    num_classes: int = 7
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    num_workers: int = 4
    seed: int = 42
    multiclass: bool = True

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
 
    def __post_init__(self) -> None:
        """Create checkpoint directory if it does not exist."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
