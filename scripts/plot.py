import os
from PIL import Image
import matplotlib.pyplot as plt
import json

def plot_image(res: tuple[object, str, str], save_path: str):
    """
    Compact layout with consistent prediction spacing
    independent of image size.
    """

    result, image_path, true_label = res
    predictions = result["predictions"][:3]

    # Load image
    img = Image.open(image_path)

    # Slightly shorter figure
    # fig = plt.figure(figsize=(5, 6))
    fig = plt.figure()

    # Tighter rows for prediction text
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[0.45, 3, 0.22, 0.22, 0.22],
        hspace=0.02
    )

    # ----- Title -----
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        f"True Label: {true_label}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold"
    )

    # ----- Image -----
    ax_img = fig.add_subplot(gs[1])
    ax_img.imshow(img)
    ax_img.axis("off")

    # ----- Prediction rows -----
    for i, pred in enumerate(predictions):
        ax_text = fig.add_subplot(gs[i + 2])
        ax_text.axis("off")

        label = pred["prediction"]
        conf = pred["confidence"]

        color = "green" if label == true_label else "red"

        ax_text.text(
            0.5,
            0.5,
            f"Top {i+1}: {label} ({conf:.3f})",
            ha="center",
            va="center",
            fontsize=11,
            color=color,
            fontweight="bold" if label == true_label else "normal"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)



def main(p):

    with open(os.path.join("out", p, "results.json"), "r") as f:
        data = json.load(f)
    

    for res in data:
        print(res[1])
        plot_image(res, os.path.join("out", p, "img", res[1].replace("custom_dataset/", "").replace(".jpg", ".png")))


if __name__ == "__main__":
    main("cnn")
    main("vit")
    main("cnn-vit")
    main("vit-cnn")