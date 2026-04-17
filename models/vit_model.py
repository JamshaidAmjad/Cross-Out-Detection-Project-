"""
vit_model.py
------------
Vision Transformer model definition for the Cross-Out Detection project.

Uses the timm library to load vit_base_patch16_224 with pretrained ImageNet
weights, then replaces the classification head for 8-class output.

Classes:
    CLEAN=0, CROSS=1, DIAGONAL=2, DOUBLE_LINE=3,
    SCRATCH=4, SINGLE_LINE=5, WAVE=6, ZIG_ZAG=7
"""

import torch
import torch.nn as nn
import timm



def get_model(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """Load vit_base_patch16_224 and replace its head for num_classes output.

    Args:
        num_classes: Number of output classes (default 8).
        pretrained: If True, load pretrained ImageNet weights via timm.

    Returns:
        A torch.nn.Module ready for training or inference.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all model parameters except the classification head.

    Call this for head-only / feature-extraction training.

    Args:
        model: The ViT model returned by get_model().
    """
    for name, param in model.named_parameters():
        if not name.startswith('head'):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning.

    Args:
        model: The ViT model returned by get_model().
    """
    for param in model.parameters():
        param.requires_grad = True