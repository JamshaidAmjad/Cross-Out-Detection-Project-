"""
transforms.py
-------------
Augmentation pipelines for the Cross-Out Detection project.

All images are resized to 224x224 as required by the Vision Transformer
(vit_base_patch16_224). Training transforms include random augmentations
for regularisation; val/test transforms are deterministic.

ImageNet normalisation stats are used throughout:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
"""

from torchvision import transforms


# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Return the augmentation pipeline for training.

    Suggested augmentations:
        - RandomResizedCrop(224)
        - RandomHorizontalFlip
        - RandomRotation (±15°)
        - ColorJitter (brightness, contrast, saturation)
        - ToTensor
        - Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Returns:
        A torchvision Compose transform.
    """
    # TODO: Build and return a transforms.Compose pipeline with the
    #       augmentations listed above
    raise NotImplementedError("Implement get_train_transforms()")


def get_val_transforms() -> transforms.Compose:
    """Return the deterministic pipeline for validation and testing.

    Steps:
        - Resize to 256
        - CenterCrop to 224
        - ToTensor
        - Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Returns:
        A torchvision Compose transform.
    """
    # TODO: Build and return a transforms.Compose pipeline with the
    #       deterministic steps listed above
    raise NotImplementedError("Implement get_val_transforms()")
