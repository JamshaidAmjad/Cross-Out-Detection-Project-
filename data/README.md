# Data Pipeline

This module handles dataset preparation, loading, and transforms.

## Expected Raw Dataset

Place or extract the dataset under `data/ImageFolder`:

```text
data/ImageFolder/
  train/images/<CLASS_NAME>/*
  val/images/<CLASS_NAME>/*
  test/images/<CLASS_NAME>/*
```

Expected raw class folders:

```text
CLEAN
CROSS
DIAGONAL
DOUBLE_LINE
SCRATCH
SINGLE_LINE
WAVE
ZIG_ZAG
MIXED
```

## Prepared Views

The dataloader uses two prepared ImageFolder views so labels stay stable:

```text
binaryclass/
  CLEAN
  MIXED

multiclass/
  CROSS
  DIAGONAL
  DOUBLE_LINE
  SCRATCH
  SINGLE_LINE
  WAVE
  ZIG_ZAG
```

On first use, `get_dataloaders()` asks whether to create these folders by copying
the raw data. The original extracted dataset is left untouched.

## Usage

```python
from data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    multiclass=True,
    data_dir="data/ImageFolder",
    batch_size=32,
    num_workers=4,
)
```
