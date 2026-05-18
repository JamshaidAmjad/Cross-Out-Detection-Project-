# Model Architecture

This module defines the Vision Transformer model for 8-class cross-out detection.

## What to Implement

### `vit_model.py`
- Use the [`timm`](https://github.com/huggingface/pytorch-image-models) library to load `vit_base_patch16_224`
- Load pretrained ImageNet weights when `pretrained=True`
- Replace the classification head to output **8 classes**
- Implement `get_model(num_classes=8, pretrained=True)` as the main factory function
- Add a `freeze_backbone(model)` utility that freezes all layers except the head
- Add an `unfreeze_backbone(model)` utility that unfreezes all layers for full fine-tuning

## Usage

```python
from models import get_model

model = get_model(num_classes=8, pretrained=True)

# For feature extraction (head-only training):
from models.vit_model import freeze_backbone
freeze_backbone(model)

# For full fine-tuning:
from models.vit_model import unfreeze_backbone
unfreeze_backbone(model)
```

## Notes
- Input images must be **224×224 RGB** (handled by the data pipeline)
- `timm` must be installed: `pip install timm`
- The default ViT head is a single `nn.Linear`; replace it with `nn.Linear(hidden_size, num_classes)`
