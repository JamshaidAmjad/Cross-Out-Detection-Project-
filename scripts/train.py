"""
scripts/train.py
----------------
Entry-point script for training the Cross-Out Detection ViT model.

Wires together the data pipeline, model, and training loop using
settings from config/config.py.

Usage:
    python scripts/train.py
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from data.dataset import get_dataloaders
from models.vit_model import get_model, freeze_backbone
from training.trainer import Trainer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cross-Out Detection ViT model")
    parser.add_argument("--data-dir", type=str, default=None, help="Root ImageFolder directory (overrides Config.data_dir)")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size (overrides Config.batch_size)")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs (overrides Config.num_epochs)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides Config.learning_rate)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ViT backbone; train classification head only")
    parser.add_argument("--binary", action="store_true", help="Train in binary mode: CLEAN vs. crossed-out (multiclass=False)")
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    # Configuration
    cfg = Config()

    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.binary:
        cfg.multiclass = False
        cfg.num_classes = 2

    print("=" * 50)
    print("Cross-Out Detection — Training")
    print("=" * 50)
    print(f"  Device       : {cfg.device}")
    print(f"  Data dir     : {cfg.data_dir}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Epochs       : {cfg.num_epochs}")
    print(f"  LR           : {cfg.learning_rate}")
    print(f"  Num classes  : {cfg.num_classes}")
    print(f"  Multiclass   : {cfg.multiclass}")
    print(f"  Checkpoints  : {cfg.checkpoint_dir}")
    print("=" * 50 + "\n")

    # Reproducibility
    set_seeds(cfg.seed)

    # Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        multiclass=cfg.multiclass,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(
        f"  Train batches : {len(train_loader)}"
        f"  |  Val batches : {len(val_loader)}"
        f"  |  Test batches: {len(test_loader)}\n"
    )

    # Model
    print("Building model (ViT-B/16, pretrained ImageNet)...")
    model = get_model(num_classes=cfg.num_classes, pretrained=True)

    if args.freeze_backbone:
        freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Backbone frozen — trainable params: {trainable:,}")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"  Full fine-tune — total params: {total:,}")
    print()

    # Training
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()

    # Done
    best_ckpt = Path(cfg.checkpoint_dir) / "best.pth"
    print(f"\nBest checkpoint saved at: {best_ckpt.resolve()}")
    print(f"Run evaluation with: python scripts/evaluate.py --checkpoint {best_ckpt}")


if __name__ == "__main__":
    main()