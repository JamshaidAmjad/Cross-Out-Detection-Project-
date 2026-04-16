"""
scripts/train.py
----------------
Entry-point script for training the Cross-Out Detection ViT model.

Wires together the data pipeline, model, and training loop using
settings from config/config.py.

Usage:
    python scripts/train.py
"""

# TODO: from config.config import Config
# TODO: from data.dataset import get_dataloaders
# TODO: from models.vit_model import get_model, freeze_backbone
# TODO: from training.trainer import Trainer


def main() -> None:
    """Run the full training pipeline."""
    # TODO: Instantiate Config
    # TODO: Set global random seeds (torch.manual_seed, random.seed, numpy seed)
    # TODO: Call get_dataloaders(cfg.data_dir, cfg.batch_size, ...) to get
    #       train_loader, val_loader, test_loader
    # TODO: Call get_model(num_classes=cfg.num_classes, pretrained=True)
    # TODO: Optionally call freeze_backbone(model) for head-only warmup
    # TODO: Instantiate Trainer(model, train_loader, val_loader, cfg)
    # TODO: Call trainer.train()
    # TODO: Print final message indicating where the best checkpoint was saved
    raise NotImplementedError("Implement main()")


if __name__ == "__main__":
    main()
