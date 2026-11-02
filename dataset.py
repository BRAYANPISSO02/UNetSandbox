import math
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SyntheticMedicalDataset(Dataset):
    """
    Generates synthetic medical-like images and binary masks on the fly.
    Simulates cellular structures or lesion targets with Gaussian noise,
    background gradients, and textured borders for benchmarking and demonstration.
    """

    def __init__(self, num_samples: int = 200, image_size: int = 128, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = w = self.image_size
        y_grid, x_grid = np.ogrid[:h, :w]

        # Simula fondo con gradientes suaves de atenuación
        bg = 0.2 + 0.1 * np.sin(x_grid / 15.0) + 0.1 * np.cos(y_grid / 15.0)
        noise = self.rng.normal(loc=0.0, scale=0.05, size=(h, w))
        image = bg + noise

        mask = np.zeros((h, w), dtype=np.float32)

        # Genera entre 1 y 4 "células" o "lesiones"
        num_targets = self.rng.integers(1, 4)
        for _ in range(num_targets):
            cx = self.rng.integers(int(w * 0.2), int(w * 0.8))
            cy = self.rng.integers(int(h * 0.2), int(h * 0.8))
            rx = self.rng.integers(int(w * 0.08), int(w * 0.18))
            ry = self.rng.integers(int(h * 0.08), int(h * 0.18))
            angle = self.rng.uniform(0, math.pi)

            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = cos_a * (x_grid - cx) + sin_a * (y_grid - cy)
            y_rot = -sin_a * (x_grid - cx) + cos_a * (y_grid - cy)

            ellipse = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0
            mask[ellipse] = 1.0

            # Realce de intensidad en la zona de la lesión
            intensity = self.rng.uniform(0.4, 0.7)
            image[ellipse] += intensity

        # Normalización min-max a [0, 1]
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return image_t, mask_t


class MedicalFolderDataset(Dataset):
    """
    Standard dataset loader for paired images and masks in directories:
        root_dir/
            images/
            masks/
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        self.transform = transform

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.filenames = sorted(
            [f.name for f in self.images_dir.iterdir() if f.suffix.lower() in valid_exts]
        )

        if not self.filenames:
            raise FileNotFoundError(f"No se encontraron imágenes válidas en {self.images_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        img_path = self.images_dir / fname
        mask_path = self.masks_dir / fname

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        img_t = torch.from_numpy(img_arr).unsqueeze(0)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        if self.transform:
            img_t, mask_t = self.transform(img_t, mask_t)

        return img_t, mask_t


def get_dataloaders(
    batch_size: int = 8,
    train_samples: int = 240,
    val_samples: int = 60,
    image_size: int = 128,
    data_dir: Optional[str] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    if data_dir and os.path.exists(data_dir):
        train_set = MedicalFolderDataset(os.path.join(data_dir, "train"))
        val_set = MedicalFolderDataset(os.path.join(data_dir, "val"))
    else:
        train_set = SyntheticMedicalDataset(train_samples, image_size=image_size, seed=42)
        val_set = SyntheticMedicalDataset(val_samples, image_size=image_size, seed=1337)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
