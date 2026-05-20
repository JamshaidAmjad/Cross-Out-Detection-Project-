from models.cnn import SimpleCNN
from torchvision import transforms
import torch
from torch import nn
from PIL import Image
from data.transforms import get_val_transforms
from models.vit_model import get_model
import torch.nn.functional as F
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# === Models ===
def cnn_binary():
    clean_model = SimpleCNN(2).to(device)
    info = torch.load(os.path.join("checkpoints/cnn/binary", "best.pth"))
    state_dict = info["model_state_dict"]
    clean_model.load_state_dict(state_dict)
    return clean_model

def cnn_multi():
    clean_model = SimpleCNN(7).to(device)
    info = torch.load(os.path.join("checkpoints/cnn/multiclass", "best.pth"))
    state_dict = info["model_state_dict"]
    clean_model.load_state_dict(state_dict)
    return clean_model

def vit_binary():
    clean_model = get_model(2).to(device)
    info = torch.load(os.path.join("checkpoints/vit/binary", "best.pth"))
    state_dict = info["model_state_dict"]
    clean_model.load_state_dict(state_dict)
    return clean_model

def vit_multi():
    clean_model = get_model(7).to(device)
    info = torch.load(os.path.join("checkpoints/vit/multiclass", "best.pth"))
    state_dict = info["model_state_dict"]
    clean_model.load_state_dict(state_dict)
    return clean_model






# === Utils ===
val_transform = get_val_transforms()
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0) # type: ignore
    return image

Multiclasses = [
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
]

def find_images(data_dir = "custom_dataset") -> list[tuple[str, str]]:
    output = []
    
    classes = os.listdir(data_dir)
    classes = [f for f in classes if os.path.isdir(os.path.join(data_dir, f))]
    
    for c in classes:
        class_dir = os.path.join(data_dir, c)
        images = os.listdir(class_dir)
        images = [f for f in images if os.path.isfile(os.path.join(class_dir, f)) and f.endswith(".jpg")]
        for image in images:
            image_path = os.path.join(class_dir, image)
            output.append((image_path, c))

    return output







# === Classifiers ==
def is_clean(model: nn.Module, image: torch.Tensor):
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)

    pred_class = torch.argmax(prediction, dim=1)
    probs = torch.softmax(prediction, dim=1)
    conf = probs.gather(1, pred_class.unsqueeze(1)).squeeze()

    return pred_class.item() == 0, conf.item(), prediction, probs

def classify(model: nn.Module, image: torch.Tensor):
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)

    pred_class = torch.argmax(prediction, dim=1)
    probs = torch.softmax(prediction, dim=1)
    conf = probs.gather(1, pred_class.unsqueeze(1)).squeeze()

    return pred_class.item(), conf.item(), prediction, probs











# === Metrics ===

def calculate_top1_score(multi_predictions, multi_confidences, label: str, classes = Multiclasses):
    """
    Returns:
        1 if top prediction matches label
        0 otherwise
    """
    pred_idx = torch.argmax(multi_confidences, dim=1).item()
    pred_label = classes[pred_idx] # type: ignore

    return int(pred_label == label)


def calculate_top3_score(multi_predictions, multi_confidences, label: str, classes = Multiclasses):
    """
    Returns:
        1 if label appears in top 3 predictions
        0 otherwise
    """
    k = min(3, multi_confidences.shape[1])
    top3_indices = torch.topk(multi_confidences, k=k, dim=1).indices[0]

    top3_labels = [classes[idx] for idx in top3_indices]

    return int(label in top3_labels)


def calculate_cross_entropy_score(multi_predictions, multi_confidences, label: str, classes = Multiclasses):
    """
    Returns cross entropy loss for a single sample.
    Lower is better.

    Uses logits (multi_predictions), NOT softmax probabilities.
    """

    target_idx = classes.index(label)

    target = torch.tensor([target_idx], device=multi_predictions.device)

    loss = F.cross_entropy(multi_predictions, target)

    return loss.item()


def calculate_mrr_score(multi_predictions, multi_confidences, label: str, classes = Multiclasses):
    """
    Mean Reciprocal Rank for a single sample.

    Returns:
        1.0 if correct class is rank 1
        0.5 if rank 2
        0.333 if rank 3
        etc.
    """
    sorted_indices = torch.argsort(
        multi_confidences,
        dim=1,
        descending=True
    )[0]

    target_idx = classes.index(label)

    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1

    return 1.0 / rank











# === Compositions ===
def calculate_clean_scores(true_positive: bool, multi_predictions: torch.Tensor, multi_confidences: torch.Tensor, classes = ["CLEAN", "MIXED"]):
    obj = {
        "scores": {
            "top1": 1 if true_positive else 0,
            "top3": 1 if true_positive else 0,
            "ce": calculate_cross_entropy_score(multi_predictions, multi_confidences, "CLEAN", classes),
            "mrr": 1 if true_positive else 0,
        },
        "predictions": []
    }

    sorted_indices = torch.argsort(
        multi_confidences,
        dim=1,
        descending=True
    )[0]

    for idx in sorted_indices:
        idx = idx.item()

        obj["predictions"].append({
            "prediction": classes[idx], # type: ignore
            "confidence": float(multi_confidences[0][idx].item())  # type: ignore
        })

    return obj


def calculate_results(multi_predictions: torch.Tensor, multi_confidences: torch.Tensor, label: str, classes = Multiclasses):
    obj = {
        "scores": {
            "top1": calculate_top1_score(multi_predictions, multi_confidences, label, classes),
            "top3": calculate_top3_score(multi_predictions, multi_confidences, label, classes),
            "ce": calculate_cross_entropy_score(multi_predictions, multi_confidences, label, classes),
            "mrr": calculate_mrr_score(multi_predictions, multi_confidences, label, classes),
        },
        "predictions": []
    }

    sorted_indices = torch.argsort(
        multi_confidences,
        dim=1,
        descending=True
    )[0]

    for idx in sorted_indices:
        idx = idx.item()

        obj["predictions"].append({
            "prediction": classes[idx], # type: ignore
            "confidence": float(multi_confidences[0][idx].item())  # type: ignore
        })

    return obj


def predict(binary: nn.Module, multi: nn.Module, image: torch.Tensor, label: str):
    pred_is_clean, clean_conf, clean_predictions, clean_confidences = is_clean(binary, image)

    if pred_is_clean:
        if label == "CLEAN": 

            # True positive
            out = "CLEAN", clean_conf, calculate_clean_scores(True, clean_predictions, clean_confidences, ["CLEAN", "MIXED"])
            return out

        # False positive
        out = "CLEAN", clean_conf, calculate_clean_scores(False, clean_predictions, clean_confidences, ["CLEAN", "MIXED"])
        return out

    elif label == "CLEAN":

        # False negative
        out = "MIXED", clean_conf, calculate_clean_scores(False, clean_predictions, clean_confidences, ["CLEAN", "MIXED"])
        return out

    
    # true negative

    type, type_conf, multi_predictions, multi_confidences = classify(multi, image)

    pred_class = Multiclasses[int(type)]

    return pred_class, type_conf, calculate_results(multi_predictions, multi_confidences, label)


def infer_scores(binary: nn.Module, multi: nn.Module, data_dir = "custom_dataset"):
    pairs = find_images(data_dir)

    results = []

    for image_path, label in pairs:

        image = load_image(image_path)
        pred, conf, out = predict(binary, multi, image, label)
        results.append((out, image_path, label))

    return results

def aggregate_scores(results):
    totals = {
        "top1": 0.0,
        "top3": 0.0,
        "ce": 0.0,
        "mrr": 0.0,
    }

    n = len(results)

    for out, _, _ in results:
        scores = out["scores"]

        totals["top1"] += scores["top1"]
        totals["top3"] += scores["top3"]
        totals["ce"] += scores["ce"]
        totals["mrr"] += scores["mrr"]

    return {
        key: value / n
        for key, value in totals.items()
    }





def main(data_dir = "custom_dataset"):
    binary = vit_binary()
    multi = vit_multi()

    results = infer_scores(binary, multi, data_dir)

    out_dir = "out"

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)
    print("Saved results to out/results.json")

    final_scores = aggregate_scores(results)

    print("\n=== FINAL SCORES ===")

    for key, value in final_scores.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()