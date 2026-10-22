"""Run the trained crosswalk and shadow segmentation model in Gradio."""

from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "best_deeplab_crosswalk_shadow_fp16.pth"
SIZE = (384, 384)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone=None, aux_loss=True
    )
    model.classifier[-1] = nn.Conv2d(256, 3, kernel_size=1)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


MODEL = load_model()


def predict(image: Image.Image) -> np.ndarray:
    if image is None:
        raise gr.Error("Upload an image first.")

    original = image.convert("RGB")
    resized = original.resize(SIZE, Image.Resampling.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), MEAN, STD).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        mask = MODEL(tensor)["out"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    mask = np.asarray(
        Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST)
    )
    overlay = np.asarray(original).copy()
    for class_id, color in ((1, (0, 255, 0)), (2, (0, 0, 255))):
        selected = mask == class_id
        overlay[selected] = (
            0.5 * overlay[selected] + 0.5 * np.asarray(color)
        ).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Road image"),
        outputs=gr.Image(type="numpy", label="Segmentation overlay"),
        title="Crosswalk and Shadow Segmentation",
        description="Crosswalk: green. Shadow: blue.",
    )
    demo.launch()
