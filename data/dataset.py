"""
dataset.py
----------
PyTorch dataset loader for the Cross-Out Detection project.

Loads images using torchvision.datasets.ImageFolder, applies class filtering
to exclude the MIXED folder, and returns train/val/test DataLoaders via
get_dataloaders().

Class mapping:
    CLEAN=0, CROSS=1, DIAGONAL=2, DOUBLE_LINE=3,
    SCRATCH=4, SINGLE_LINE=5, WAVE=6, ZIG_ZAG=7
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from data.transforms import get_train_transforms, get_val_transforms

# Classes to include (MIXED is excluded)
CLASS_NAMES = [
    "CLEAN",
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return train, val, and test DataLoaders.

    Args:
        data_dir: Root directory containing per-class subfolders.
        batch_size: Number of samples per batch.
        val_split: Fraction of data used for validation.
        test_split: Fraction of data used for testing.
        num_workers: Number of worker processes for data loading.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # TODO: Load full dataset with ImageFolder using get_train_transforms()
    # TODO: Filter out the MIXED class using a custom is_valid_file or
    #       by overriding the class-to-index mapping
    # TODO: Split indices into train / val / test using torch.randperm or
    #       sklearn.model_selection.train_test_split with the given seed
    # TODO: Create Subset datasets for each split
    # TODO: Apply get_val_transforms() to val and test subsets
    # TODO: Wrap each Subset in a DataLoader with appropriate shuffle settings
    #       (shuffle=True for train, False for val/test)
    raise NotImplementedError("Implement get_dataloaders()")
