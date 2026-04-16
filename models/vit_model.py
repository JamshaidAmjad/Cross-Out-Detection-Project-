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

# TODO: import timm


def get_model(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """Load vit_base_patch16_224 and replace its head for num_classes output.

    Args:
        num_classes: Number of output classes (default 8).
        pretrained: If True, load pretrained ImageNet weights via timm.

    Returns:
        A torch.nn.Module ready for training or inference.
    """
    # TODO: Use timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    # TODO: Retrieve the input feature size of the existing head (model.head.in_features)
    # TODO: Replace model.head with nn.Linear(in_features, num_classes)
    # TODO: Return the modified model
    raise NotImplementedError("Implement get_model()")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all model parameters except the classification head.

    Call this for head-only / feature-extraction training.

    Args:
        model: The ViT model returned by get_model().
    """
    # TODO: Iterate over all named parameters
    # TODO: Set requires_grad=False for every parameter whose name does
    #       not start with 'head'
    raise NotImplementedError("Implement freeze_backbone()")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning.

    Args:
        model: The ViT model returned by get_model().
    """
    # TODO: Iterate over all parameters and set requires_grad=True
    raise NotImplementedError("Implement unfreeze_backbone()")
