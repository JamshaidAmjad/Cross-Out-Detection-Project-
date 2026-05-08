from models.cnn import SimpleCNN
from torchvision import transforms
import torch
from PIL import Image
from data.transforms import get_val_transforms
import os

BINARY_CLASSIFIER_PATH = "checkpoints_binary"
MULTI_CLASSIFIER_PATH = "checkpoints"

def is_clean(image: torch.Tensor, model: torch.nn.Module):
    device = next(iter(model.parameters())).device

    model.eval()

    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)

    pred_class = torch.argmax(prediction, dim=1)
    probs = torch.softmax(prediction, dim=1)
    conf = probs.gather(1, pred_class.unsqueeze(1)).squeeze()

    return pred_class.item() == 0, conf.item()



val_transform = get_val_transforms()

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0) # type: ignore

    return image

def find_type(image: torch.Tensor, model: torch.nn.Module):
    device = next(iter(model.parameters())).device

    model.eval()

    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)

    pred_class = torch.argmax(prediction, dim=1)
    probs = torch.softmax(prediction, dim=1)
    conf = probs.gather(1, pred_class.unsqueeze(1)).squeeze()

    return pred_class.item(), conf.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clean_model = SimpleCNN(2).to(device)
info = torch.load(os.path.join(BINARY_CLASSIFIER_PATH, "best.pth"))
state_dict = info["model_state_dict"]
clean_model.load_state_dict(state_dict)

type_model = SimpleCNN(8).to(device)
info = torch.load(os.path.join(MULTI_CLASSIFIER_PATH, "best.pth"))
state_dict = info["model_state_dict"]
type_model.load_state_dict(state_dict)

Multiclasses = [
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
]

def predict(img:torch.Tensor) -> tuple[str, float]:
    clean, clean_conf = is_clean(img, clean_model)

    if clean:
        return "CLEAN", clean_conf

    pred, pred_conf = find_type(img, type_model)
    
    pred_class = Multiclasses[int(pred)]

    return pred_class, pred_conf

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


def main(data_dir = "custom_dataset"):
    pairs = find_images(data_dir)

    success = 0
    total = 0

    for image_path, label in pairs:
        image = load_image(image_path)
        pred, conf = predict(image)

        if pred == label:
            success += 1
            print(f"Correct {pred} with {conf*100:.2f}%")
        elif pred == "CLEAN":
            pred_2, pred_conf = find_type(image, type_model)

            p = Multiclasses[int(pred_2)]

            correct = p == label

            if correct:
                print(f"If not CLEAN {conf*100:.2f}%: Correct {p} with {pred_conf*100:.2f}%")
            else:
                print(f"If not CLEAN {conf*100:.2f}%: {p} but was {label} with {pred_conf*100:.2f}%")
        else:
            print(f"{pred} but was {label} with {conf*100:.2f}%")


        total += 1

    
    print(f"{success} / {total}")

if __name__ == "__main__":
    main()