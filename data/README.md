# Data Pipeline

This module handles all dataset loading, preprocessing, and augmentation for the Cross-Out Detection project.

## What to Implement

### `dataset.py`
- Create a PyTorch `ImageFolder`-based dataset loader
- Implement `get_dataloaders()` function that returns train, val, and test `DataLoader` objects
- Apply the class mapping below (exclude the `MIXED` folder entirely):

| Class | Label |
|-------|-------|
| CLEAN | 0 |
| CROSS | 1 |
| DIAGONAL | 2 |
| DOUBLE_LINE | 3 |
| SCRATCH | 4 |
| SINGLE_LINE | 5 |
| WAVE | 6 |
| ZIG_ZAG | 7 |

- Use `torchvision.datasets.ImageFolder` with a custom class filter to exclude `MIXED`
- Split into train / val / test sets (suggested: 70 / 15 / 15)

### `transforms.py`
- Implement augmentation pipelines for training and validation/test
- All images must be resized to **224×224** (required by ViT)
- Training augmentations should include: random horizontal flip, random rotation, color jitter, normalization (ImageNet mean/std)
- Validation/test transforms: resize + center crop + normalization only

## Usage

```python
from data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir="path/to/dataset",
    batch_size=32,
)
```
