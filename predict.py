import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import SyntheticMedicalDataset
from model import UNet


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    model = UNet(in_channels=1, num_classes=1, base_c=32).to(device)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[+] Pesos cargados desde: {checkpoint_path}")
    else:
        print(f"[!] Aviso: No se encontró {checkpoint_path}. Se usará el modelo sin entrenar (pesos aleatorios).")
    model.eval()
    return model


def predict_sample(
    model: UNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
):
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device) if image_tensor.dim() == 3 else image_tensor.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probs > threshold).astype(np.float32)
    return probs, pred_mask


def main():
    parser = argparse.ArgumentParser(description="Inferencia y visualización con U-Net")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Ruta al checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Ruta a una imagen en escala de grises opcional")
    parser.add_argument("--output", type=str, default="prediction_sample.png", help="Ruta para guardar el plot")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de binarización")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.image and Path(args.image).exists():
        raw_img = Image.open(args.image).convert("L")
        img_arr = np.array(raw_img, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(img_arr).unsqueeze(0)
        gt_mask = None
    else:
        dataset = SyntheticMedicalDataset(num_samples=1, image_size=128, seed=999)
        image_t, mask_t = dataset[0]
        gt_mask = mask_t.squeeze().numpy()

    probs, pred_mask = predict_sample(model, image_t, device, threshold=args.threshold)
    input_img = image_t.squeeze().numpy()

    cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    axes[0].imshow(input_img, cmap="bone")
    axes[0].set_title("Entrada (Input)")
    axes[0].axis("off")

    col_idx = 1
    if gt_mask is not None:
        axes[col_idx].imshow(gt_mask, cmap="gray")
        axes[col_idx].set_title("Ground Truth")
        axes[col_idx].axis("off")
        col_idx += 1

        # Métrica puntual
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
        print(f"[+] Dice score sobre la muestra: {dice:.4f}")

    axes[col_idx].imshow(probs, cmap="magma")
    axes[col_idx].set_title("Mapa de Probabilidad")
    axes[col_idx].axis("off")
    col_idx += 1

    axes[col_idx].imshow(pred_mask, cmap="gray")
    axes[col_idx].set_title(f"Predicción (Umbral > {args.threshold})")
    axes[col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Visualización guardada en: {args.output}")


if __name__ == "__main__":
    main()
