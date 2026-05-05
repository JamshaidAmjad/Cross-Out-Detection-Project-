# Training Pipeline

This module implements the training loop and all supporting utilities for the Cross-Out Detection project.

## What to Implement

### `trainer.py`

Implement a `Trainer` class (or a `train()` function) that covers:

| Component | Details |
|-----------|---------|
| **Loss** | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| **Optimiser** | `torch.optim.AdamW(lr=cfg.learning_rate, weight_decay=cfg.weight_decay)` |
| **Scheduler** | `CosineAnnealingLR(T_max=cfg.num_epochs)` |
| **Early stopping** | Stop if val loss does not improve for `cfg.early_stopping_patience` epochs |
| **Checkpointing** | Save best model to `cfg.checkpoint_dir/best.pth`; save latest to `last.pth` |
| **Metrics** | Log train loss, val loss, and val accuracy each epoch |

### Training loop (per epoch)
1. Set model to `train()` mode; iterate over `train_loader`
2. Forward pass → compute loss → backward → optimiser step
3. Set model to `eval()` mode; iterate over `val_loader` with `torch.no_grad()`
4. Compute val loss and accuracy
5. Step scheduler; check early stopping condition
6. Save checkpoint if val loss improved

## Usage

```python
from training.trainer import Trainer
from config import Config

cfg = Config()
trainer = Trainer(model, train_loader, val_loader, cfg)
trainer.train()
```
