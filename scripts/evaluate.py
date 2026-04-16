"""
scripts/evaluate.py
-------------------
Entry-point script for evaluating a trained Cross-Out Detection model.

Loads a saved checkpoint, runs inference on the test set, and reports
per-class and macro metrics, binary accuracy, and saves visualisations.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth
"""

import argparse

# TODO: from config.config import Config
# TODO: from data.dataset import get_dataloaders
# TODO: from models.vit_model import get_model
# TODO: from evaluation.metrics import (
#     compute_accuracy, compute_per_class_metrics,
#     compute_macro_metrics, compute_binary_accuracy,
# )
# TODO: from evaluation.visualize import plot_confusion_matrix, plot_training_curves
# TODO: import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Cross-Out Detection model")
    # TODO: Add --checkpoint argument (required, path to .pth file)
    # TODO: Add --data-dir argument (optional, overrides cfg.data_dir)
    # TODO: Add --output-dir argument (optional, where to save plots)
    raise NotImplementedError("Implement parse_args()")


def main() -> None:
    """Run the evaluation pipeline."""
    # TODO: Call parse_args()
    # TODO: Instantiate Config; override fields from args if provided
    # TODO: Call get_dataloaders to obtain test_loader
    # TODO: Call get_model(num_classes=cfg.num_classes, pretrained=False)
    # TODO: Load checkpoint weights with torch.load and model.load_state_dict
    # TODO: Move model to cfg.device and set to eval mode
    # TODO: Run inference loop over test_loader with torch.no_grad()
    #       collecting all preds and labels
    # TODO: Call compute_accuracy, compute_per_class_metrics,
    #       compute_macro_metrics, compute_binary_accuracy and print results
    # TODO: Call plot_confusion_matrix and save to output_dir
    raise NotImplementedError("Implement main()")


if __name__ == "__main__":
    main()
