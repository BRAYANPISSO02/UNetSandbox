import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_area = (preds_flat.sum() + targets_flat.sum()).item()
    union = total_area - intersection

    dice = (2.0 * intersection + eps) / (total_area + eps)
    iou = (intersection + eps) / (union + eps)

    return {"dice": dice, "iou": iou}


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        metrics = compute_metrics(logits, masks)
        total_dice += metrics["dice"] * images.size(0)
        total_iou += metrics["iou"] * images.size(0)

    n = len(dataloader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de U-Net en segmentación biomédica")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Resolución de las imágenes")
    parser.add_argument("--data-dir", type=str, default=None, help="Directorio con datos reales")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Ruta de guardado")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Dispositivo seleccionado: {device}")

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir,
    )

    model = UNet(in_channels=1, num_classes=1, base_c=32).to(device)
    criterion = BCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_path = Path(args.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    best_model_file = ckpt_path / "best_model.pth"

    print("=" * 60)
    print(f"Iniciando entrenamiento ({args.epochs} épocas)")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Época {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                best_model_file,
            )

    print("=" * 60)
    print(f"Entrenamiento finalizado. Mejor Val Dice: {best_val_dice:.4f}")
    print(f"Modelo guardado en: {best_model_file}")


if __name__ == "__main__":
    main()
