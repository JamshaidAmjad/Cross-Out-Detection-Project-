import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
from vit_model import get_model
# from data.transforms import get_train_transforms, get_val_transforms : to add when merged with data-pipeline branch


# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load an image from disk and apply the same preprocessing as training.

    Args:
        image_path: Path to the input image.
        
    Returns:
        A preprocessed image tensor ready for model input.
    """
    
    # To be replaced with get_val_transforms() when merged with data-pipeline branch
    transform = Compose([ transforms.Resize((224,224)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def start_inference(model_path: str | None = None) -> None:
    """Start an interactive inference loop using ViT.

    Args:
        model_path: Optional checkpoint path. If omitted, inference uses only
            pretrained ImageNet weights from get_model(pretrained=True).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=8, pretrained=True)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {model_path}")
    else:
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
        print(f"Predicted class: {pred_class}")
        
if __name__ == "__main__":
    try:
        start_inference(sys.argv[1] if len(sys.argv) > 1 else None)
    except KeyboardInterrupt:
        print("\nQuitting....")