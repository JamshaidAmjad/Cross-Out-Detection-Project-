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
- The dataset given form canvas already partions our data into test, val and train.

### `transforms.py`
- Implement augmentation pipelines for training and validation/test
- All images must be resized to **224×224** (required by ViT)
- Training augmentations should include: random horizontal flip, random rotation, color jitter, normalization (ImageNet mean/std)
- Validation/test transforms: resize + center crop + normalization only

## Usage

Setup:
In the folder "data" create a file called "ImageFolder". 
In "ImageFolder" simply exctract the dataset given in the instructions from canvas.
If everything works correctly, and you run from a file outside the "data" folder, you will not need to touch the "data_dir".


```python
from data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir="path/to/dataset",
    batch_size=32,
)
```
