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
from torch.utils.data import DataLoader

# TODO: from config.config import Config


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
        cfg,  # TODO: type as Config
    ) -> None:
        # TODO: Store model, loaders, and cfg as instance attributes
        # TODO: Move model to cfg.device
        # TODO: Initialise loss: nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        # TODO: Initialise optimiser: torch.optim.AdamW(...)
        # TODO: Initialise scheduler: CosineAnnealingLR(optimiser, T_max=cfg.num_epochs)
        # TODO: Initialise early-stopping counter and best_val_loss tracker
        # TODO: Create cfg.checkpoint_dir if it does not exist
        raise NotImplementedError("Implement Trainer.__init__()")

    def train(self) -> None:
        """Run the full training loop for cfg.num_epochs epochs."""
        # TODO: Loop over range(cfg.num_epochs)
        # TODO: Call self._train_epoch() and self._val_epoch() each iteration
        # TODO: Step the scheduler
        # TODO: Check early stopping condition
        # TODO: Save checkpoint if val_loss improved
        raise NotImplementedError("Implement Trainer.train()")

    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (average train loss, train accuracy).
        """
        # TODO: Set model to train mode
        # TODO: Iterate over train_loader
        # TODO: Zero gradients, forward pass, compute loss, backward, step
        # TODO: Accumulate loss and correct predictions
        # TODO: Return mean loss and accuracy
        raise NotImplementedError("Implement Trainer._train_epoch()")

    def _val_epoch(self) -> Tuple[float, float]:
        """Run one validation epoch.

        Returns:
            Tuple of (average val loss, val accuracy).
        """
        # TODO: Set model to eval mode
        # TODO: Use torch.no_grad() context
        # TODO: Iterate over val_loader
        # TODO: Compute loss and accuracy
        # TODO: Return mean loss and accuracy
        raise NotImplementedError("Implement Trainer._val_epoch()")

    def _save_checkpoint(self, filename: str) -> None:
        """Save model state dict to cfg.checkpoint_dir/filename.

        Args:
            filename: Name of the checkpoint file (e.g. 'best.pth').
        """
        # TODO: Build full path from cfg.checkpoint_dir / filename
        # TODO: Save model.state_dict() with torch.save
        raise NotImplementedError("Implement Trainer._save_checkpoint()")
