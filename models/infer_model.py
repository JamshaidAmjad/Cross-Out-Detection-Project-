import sys
from pathlib import Path

import torch
from PIL import Image

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.transforms import get_val_transforms
from models.vit_model import get_model


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load an image from disk and apply the same preprocessing as training.

    Args:
        image_path: Path to the input image.
        
    Returns:
        A preprocessed image tensor ready for model input.
    """
    
    transform = get_val_transforms()
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def start_inference(model_path: str | None = None) -> None:
    """Start an interactive inference loop using ViT.

    Args:
        model_path: Optional checkpoint path. If omitted, inference uses only
            pretrained ImageNet weights from get_model(pretrained=True).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 8
    class_names = None
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            num_classes = checkpoint.get("num_classes", num_classes)
            class_names = checkpoint.get("class_names")
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint
        model = get_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {model_path}")
    else:
        model = get_model(num_classes=num_classes, pretrained=True)
        print("No checkpoint provided. Using pretrained ImageNet weights only.")

    model = model.to(device)
    model.eval()

    print("\nViT model ready. Enter image paths for inference (type 'exit' to quit).")

    while True:
        img_path = input("> ").strip()
        if img_path.lower() in ["exit", "quit"]:
            print("Exiting inference loop.")
            break

        if not Path(img_path).is_file():
            print(f"Invalid path: {img_path}")
            continue

        image = load_and_preprocess_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
        pred_class = torch.argmax(output, dim=1).item()
        pred_label = class_names[pred_class] if class_names else str(pred_class)
        print(f"Predicted class: {pred_label}")
        
if __name__ == "__main__":
    try:
        start_inference(sys.argv[1] if len(sys.argv) > 1 else None)
    except KeyboardInterrupt:
        print("\nQuitting....")
