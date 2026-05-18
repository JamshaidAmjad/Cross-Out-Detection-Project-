# Configuration

This module centralises all hyperparameters and path settings for the Cross-Out Detection project.

## What to Implement

### `config.py`
Define a `Config` dataclass (or plain class) with at least the following fields:

| Field | Suggested default | Description |
|-------|------------------|-------------|
| `data_dir` | `"data/dataset"` | Root directory of the image dataset |
| `checkpoint_dir` | `"checkpoints/"` | Where to save model checkpoints |
| `num_classes` | `7` | Number of output classes for the selected mode |
| `image_size` | `224` | Input resolution for ViT |
| `batch_size` | `32` | Samples per mini-batch |
| `num_epochs` | `50` | Maximum training epochs |
| `learning_rate` | `1e-4` | Initial learning rate for AdamW |
| `weight_decay` | `1e-2` | L2 regularisation for AdamW |
| `label_smoothing` | `0.1` | Label smoothing for CrossEntropyLoss |
| `early_stopping_patience` | `10` | Epochs without improvement before stopping |
| `device` | auto-detect | `"cuda"` if available, else `"cpu"` |
| `num_workers` | `4` | DataLoader worker processes |
| `seed` | `42` | Global random seed |

## Usage

```python
from config import Config

cfg = Config()
print(cfg.device)   # "cuda" or "cpu"
print(cfg.learning_rate)  # 1e-4
```
