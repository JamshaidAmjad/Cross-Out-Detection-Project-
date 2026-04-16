"""
scripts/inference.py
--------------------
Single-image inference script for the Cross-Out Detection model.

Loads a trained checkpoint, preprocesses one image, runs a forward pass,
and prints the predicted class name and confidence score.

Usage:
    python scripts/inference.py --image path/to/image.jpg \
                                 --checkpoint checkpoints/best.pth
"""

import argparse
from pathlib import Path

# TODO: import torch
# TODO: from PIL import Image
# TODO: from config.config import Config
# TODO: from models.vit_model import get_model
# TODO: from data.transforms import get_val_transforms
# TODO: from data.dataset import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    # TODO: Add --image argument (required, path to input image)
    # TODO: Add --checkpoint argument (required, path to .pth file)
    # TODO: Add --top-k argument (optional int, default=3, show top-k predictions)
    raise NotImplementedError("Implement parse_args()")


def predict(image_path: str, checkpoint_path: str, top_k: int = 3) -> None:
    """Load model and run inference on a single image.

    Args:
        image_path: Path to the input image file.
        checkpoint_path: Path to the saved model checkpoint (.pth).
        top_k: Number of top predictions to display.
    """
    # TODO: Instantiate Config
    # TODO: Call get_model(num_classes=cfg.num_classes, pretrained=False)
    # TODO: Load checkpoint with torch.load and model.load_state_dict
    # TODO: Move model to cfg.device and set to eval mode
    # TODO: Open image with PIL.Image.open and convert to RGB
    # TODO: Apply get_val_transforms() to the image; add batch dimension (unsqueeze(0))
    # TODO: Move tensor to cfg.device
    # TODO: Run forward pass inside torch.no_grad()
    # TODO: Apply softmax to get class probabilities
    # TODO: Get top_k indices and scores with torch.topk
    # TODO: Print each top-k prediction: class name and confidence percentage
    raise NotImplementedError("Implement predict()")


def main() -> None:
    """Entry point."""
    # TODO: Call parse_args()
    # TODO: Call predict(args.image, args.checkpoint, args.top_k)
    raise NotImplementedError("Implement main()")


if __name__ == "__main__":
    main()
