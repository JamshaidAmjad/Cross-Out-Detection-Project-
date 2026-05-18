"""
scripts/evaluate.py
-------------------
Evaluate a trained Cross-Out Detection checkpoint on the test split.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config
from data.dataset import get_class_names, get_dataloaders
from evaluation.metrics import (
    compute_accuracy,
    compute_binary_accuracy,
    compute_per_class_metrics,
)
from evaluation.visualize import plot_confusion_matrix, plot_training_curves
from models.vit_model import get_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Cross-Out Detection model")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", type=str, default=None, help="Root ImageFolder directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs", help="Where to save metrics and plots")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--binary", action="store_true", help="Evaluate binary CLEAN vs MIXED model")
    parser.add_argument("--no-plots", action="store_true", help="Skip confusion matrix and curve plots")
    return parser.parse_args()


def _checkpoint_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _class_names_for_report(
    checkpoint: object,
    dataset_class_names: list[str],
    num_classes: int,
) -> list[str]:
    if isinstance(checkpoint, dict):
        checkpoint_class_names = checkpoint.get("class_names")
        if checkpoint_class_names and len(checkpoint_class_names) == num_classes:
            return list(checkpoint_class_names)

    class_names = list(dataset_class_names)
    while len(class_names) < num_classes:
        class_names.append(f"unused_{len(class_names)}")
    return class_names[:num_classes]


def main() -> None:
    """Run the evaluation pipeline."""
    args = parse_args()
    cfg = Config()

    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    cfg.multiclass = not args.binary
    cfg.num_classes = len(get_class_names(cfg.multiclass))

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    _, _, test_loader = get_dataloaders(
        multiclass=cfg.multiclass,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dataset_class_names = test_loader.dataset.classes

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    checkpoint_num_classes = checkpoint.get("num_classes") if isinstance(checkpoint, dict) else None
    num_classes = int(checkpoint_num_classes or cfg.num_classes)
    class_names = _class_names_for_report(checkpoint, dataset_class_names, num_classes)

    if num_classes != len(dataset_class_names):
        print(
            "Warning: checkpoint output classes do not match dataset classes "
            f"({num_classes} vs {len(dataset_class_names)}). Evaluation will still run."
        )

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        progress = tqdm(test_loader, desc="evaluate", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = compute_accuracy(all_preds, all_labels)
    per_class = compute_per_class_metrics(all_preds, all_labels, class_names)
    macro = {
        "precision": sum(metrics["precision"] for metrics in per_class.values()) / len(per_class),
        "recall": sum(metrics["recall"] for metrics in per_class.values()) / len(per_class),
        "f1": sum(metrics["f1"] for metrics in per_class.values()) / len(per_class),
    }

    results = {
        "checkpoint": str(checkpoint_path),
        "data_dir": cfg.data_dir,
        "mode": "multiclass" if cfg.multiclass else "binary",
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": accuracy,
        "macro": macro,
        "per_class": per_class,
    }

    if "CLEAN" in class_names:
        clean_idx = class_names.index("CLEAN")
        results["binary_clean_vs_other_accuracy"] = compute_binary_accuracy(
            all_preds,
            all_labels,
            clean_class_idx=clean_idx,
        )

    print("\nEvaluation results")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro P  : {macro['precision']:.4f}")
    print(f"Macro R  : {macro['recall']:.4f}")
    print(f"Macro F1 : {macro['f1']:.4f}")
    print("\nPer-class metrics")
    for class_name, metrics in per_class.items():
        print(
            f"  {class_name:<12} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"N={metrics['support']}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics: {metrics_path}")

    if not args.no_plots:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(all_preds, all_labels, class_names, save_path=str(cm_path))
        print(f"Saved confusion matrix: {cm_path}")

        if isinstance(checkpoint, dict) and checkpoint.get("history"):
            history = checkpoint["history"]
            curves_path = output_dir / "training_curves.png"
            plot_training_curves(
                history.get("train_loss", []),
                history.get("val_loss", []),
                history.get("val_accuracy", []),
                save_path=str(curves_path),
            )
            print(f"Saved training curves: {curves_path}")


if __name__ == "__main__":
    main()
