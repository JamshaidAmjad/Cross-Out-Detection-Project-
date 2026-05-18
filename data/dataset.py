"""
dataset.py
----------
PyTorch dataset loader for the Cross-Out Detection project.

The raw dataset is expected in this shape:
    data/ImageFolder/train/images/<CLASS_NAME>/*
    data/ImageFolder/val/images/<CLASS_NAME>/*
    data/ImageFolder/test/images/<CLASS_NAME>/*

For stability with torchvision.datasets.ImageFolder, setup_data() copies the
raw data into two prepared views:
    binaryclass: CLEAN vs MIXED
    multiclass: 7 cross-out style classes
"""

from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import get_train_transforms, get_val_transforms


BINARY_CLASS_NAMES = ["CLEAN", "MIXED"]

MULTICLASS_CLASS_NAMES = [
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
]

CLASS_NAMES = ["CLEAN", *MULTICLASS_CLASS_NAMES, "MIXED"]
SPLITS = ["train", "val", "test"]


def get_class_names(multiclass: bool) -> list[str]:
    """Return expected class names for the selected training mode."""
    return MULTICLASS_CLASS_NAMES if multiclass else BINARY_CLASS_NAMES


def _prepared_root(data_dir: str | Path, multiclass: bool) -> Path:
    mode_dir = "multiclass" if multiclass else "binaryclass"
    return Path(data_dir) / mode_dir


def _split_images_root(data_dir: str | Path, multiclass: bool, split: str) -> Path:
    return _prepared_root(data_dir, multiclass) / split / "images"


def _prepared_data_exists(data_dir: str | Path) -> bool:
    for multiclass in [True, False]:
        for split in SPLITS:
            split_root = _split_images_root(data_dir, multiclass, split)
            if not split_root.is_dir():
                return False
            for class_name in get_class_names(multiclass):
                if not (split_root / class_name).is_dir():
                    return False
    return True


def setup_data(dirpath: str | Path) -> None:
    """Copy the raw ImageFolder data into binary and multiclass views."""
    data_root = Path(dirpath)
    mode_specs = {
        "multiclass": MULTICLASS_CLASS_NAMES,
        "binaryclass": BINARY_CLASS_NAMES,
    }

    for split_index, split in enumerate(SPLITS, start=1):
        print(f"copying set {split} ({split_index}/{len(SPLITS)})")
        source_split_root = data_root / split / "images"
        if not source_split_root.is_dir():
            raise FileNotFoundError(f"Missing source split folder: {source_split_root}")

        for mode_name, class_names in mode_specs.items():
            target_split_root = data_root / mode_name / split / "images"
            target_split_root.mkdir(parents=True, exist_ok=True)

            for class_index, class_name in enumerate(class_names, start=1):
                print(f"{mode_name}: ({class_index}/{len(class_names)}) {class_name}")
                source = source_split_root / class_name
                target = target_split_root / class_name
                if not source.is_dir():
                    raise FileNotFoundError(f"Missing class folder: {source}")
                shutil.copytree(source, target, dirs_exist_ok=True)


def get_dataloaders(
    multiclass: bool,
    data_dir: str = "data/ImageFolder",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Return train, validation, and test dataloaders for the selected mode."""
    if not _prepared_data_exists(data_dir):
        inp = input(
            "Data folders have to be formatted before use. "
            "Create binaryclass/ and multiclass/ copies now? (y/n) "
        )
        if inp.lower() == "y":
            setup_data(dirpath=data_dir)
        else:
            raise RuntimeError("Prepared data folders are required for training.")

    train_dataset = ImageFolder(
        root=_split_images_root(data_dir, multiclass, "train"),
        transform=get_train_transforms(),
    )
    val_dataset = ImageFolder(
        root=_split_images_root(data_dir, multiclass, "val"),
        transform=get_val_transforms(),
    )
    test_dataset = ImageFolder(
        root=_split_images_root(data_dir, multiclass, "test"),
        transform=get_val_transforms(),
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
