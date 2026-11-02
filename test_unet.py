import unittest
import torch

from model import UNet
from dataset import SyntheticMedicalDataset
from train import DiceLoss, BCEDiceLoss, compute_metrics


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=1, num_classes=1, base_c=16).to(self.device)

    def test_forward_shape_standard(self):
        x = torch.randn(2, 1, 128, 128, device=self.device)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_forward_shape_arbitrary(self):
        # Prueba con resoluciones no múltiplos exactos de 32 para validar el padding dinámico
        x = torch.randn(2, 1, 130, 130, device=self.device)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 1, 130, 130))

    def test_gradient_flow(self):
        x = torch.randn(2, 1, 64, 64, device=self.device)
        target = torch.ones(2, 1, 64, 64, device=self.device)
        criterion = BCEDiceLoss()

        out = self.model(x)
        loss = criterion(out, target)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradiente None en {name}")

    def test_dice_loss_bounds(self):
        criterion = DiceLoss()
        target = torch.ones(1, 1, 32, 32)

        # Logits altos positivos -> prob ~ 1 -> Dice ~ 1 -> Loss ~ 0
        perfect_logits = torch.full((1, 1, 32, 32), 15.0)
        loss_perfect = criterion(perfect_logits, target).item()
        self.assertLess(loss_perfect, 0.05)

        # Logits muy negativos -> prob ~ 0 -> Dice ~ 0 -> Loss ~ 1
        worst_logits = torch.full((1, 1, 32, 32), -15.0)
        loss_worst = criterion(worst_logits, target).item()
        self.assertGreater(loss_worst, 0.95)

    def test_synthetic_dataset(self):
        ds = SyntheticMedicalDataset(num_samples=5, image_size=64, seed=42)
        self.assertEqual(len(ds), 5)
        img, mask = ds[0]
        self.assertEqual(img.shape, (1, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))
        self.assertTrue((img >= 0.0).all() and (img <= 1.0).all())
        self.assertTrue(set(torch.unique(mask).tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
