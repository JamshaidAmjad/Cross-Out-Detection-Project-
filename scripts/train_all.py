import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from data.dataset import get_dataloaders
from models.vit_model import get_model, freeze_backbone
from training.trainer import Trainer
from models.cnn import SimpleCNN

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model: nn.Module, cfg: Config, model_name: str):
    print("=" * 50)
    print("Cross-Out Detection — Training")
    print("=" * 50)
    print(f"  Model        : {model_name}")
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

    print(f"  Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    trainer = Trainer(model, train_loader, val_loader, cfg, model_name)
    trainer.train()

    best_ckpt = Path(cfg.checkpoint_dir) / "best.pth"
    print(f"\nBest checkpoint saved at: {best_ckpt.resolve()}")
    print(f"Run evaluation with: python scripts/evaluate.py --checkpoint {best_ckpt}")


def cnn_binary(cfg: Config, data_dir: str = "data/ImageFolder/"):
    cfg.multiclass = False
    cfg.data_dir = data_dir
    cfg.num_classes = 2

    model = SimpleCNN(2)
    train_model(model, cfg, "cnn")

def cnn_multi(cfg: Config, data_dir: str = "data/ImageFolder/"):
    cfg.multiclass = True
    cfg.data_dir = data_dir
    cfg.num_classes = 7

    model = SimpleCNN(7)
    train_model(model, cfg, "cnn")

def vit_binary(cfg: Config, freeze: bool, data_dir: str = "data/ImageFolder/"):
    cfg.multiclass = False
    cfg.data_dir = data_dir
    cfg.num_classes = 2

    model = get_model(2, True)

    if freeze:
        freeze_backbone(model)

    train_model(model, cfg, "vit")

def vit_multi(cfg: Config, freeze: bool, data_dir: str = "data/ImageFolder/"):
    cfg.multiclass = True
    cfg.data_dir = data_dir
    cfg.num_classes = 7

    model = get_model(7, True)

    if freeze:
        freeze_backbone(model)

    train_model(model, cfg, "vit")


def main():
    num_epochs = 2

    cnn_binary(Config(
        num_epochs=num_epochs
    ))
    cnn_multi(Config(
        num_epochs=num_epochs
    ))
    vit_binary(Config(
        num_epochs=num_epochs
    ), True)
    vit_multi(Config(
        num_epochs=num_epochs
    ), True)

if __name__ == "__main__":
    main()