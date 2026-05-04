"""
trainer.py
----------
Training loop for the Cross-Out Detection ViT model.

Implements the Trainer class which handles:
- CrossEntropyLoss with label smoothing
- AdamW optimiser
- CosineAnnealingLR scheduler
- Early stopping based on validation loss
- Best/latest checkpoint saving
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config.config import Config


class Trainer:
    """Manages the full training and validation loop.

    Args:
        model: The ViT model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        cfg: Config instance with all hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # Optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler — cosine decay over all epochs
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=cfg.num_epochs)

        # Early stopping state
        self._patience_counter: int = 0
        self._best_val_loss: float = float("inf")

        # Checkpoint directory
        self._ckpt_dir = Path(cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the full training loop for cfg.num_epochs epochs."""
        print(
            f"Training on {self.cfg.device.upper()} "
            f"for up to {self.cfg.num_epochs} epochs "
            f"(patience={self.cfg.early_stopping_patience})\n"
            f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>8}"
        )
        print("-" * 42)

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()

            self.scheduler.step()

            print(
                f"{epoch:>6} | {train_loss:>10.4f} | "
                f"{val_loss:>8.4f} | {val_acc:>7.2%}"
            )

            # Save the latest checkpoint
            self._save_checkpoint("last.pth")

            # Save best and check early stopping
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint("best.pth")
                print(f"         ✓ Best val loss improved → saved best.pth")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.cfg.early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered after {epoch} epochs "
                        f"(no improvement for {self.cfg.early_stopping_patience} epochs)."
                    )
                    break

        print(
            f"\nTraining complete. "
            f"Best val loss: {self._best_val_loss:.4f}. "
            f"Checkpoints saved to '{self.cfg.checkpoint_dir}'."
        )

    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (average train loss, train accuracy).
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimiser.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _val_epoch(self) -> Tuple[float, float]:
        """Run one validation epoch.

        Returns:
            Tuple of (average val loss, val accuracy).
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str) -> None:
        """Save model state dict to cfg.checkpoint_dir/filename.

        Args:
            filename: Name of the checkpoint file (e.g. 'best.pth').
        """
        save_path = self._ckpt_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.cfg.num_classes,
                "best_val_loss": self._best_val_loss,
            },
            save_path,
        )