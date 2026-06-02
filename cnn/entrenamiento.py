"""
cnn/entrenamiento.py
Entrena la CNN (EMNIST Letters A-Z + MNIST 0-9) con augmentation agresiva
para cerrar el domain gap con placas reales capturadas por cámara.
Guarda: models/modelo_entrenado.pt
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from modelo import crear_modelo_cnn, NUM_CLASES

torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
RUTA_MODELO = str(ROOT / "models" / "modelo_entrenado.pt")
EPOCHS = 25
BATCH_SIZE = 256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------
#  Datasets
# ----------------------------------------------------------------

class MNISTOffsetDataset(Dataset):
    """MNIST 0-9 con etiquetas 26-35."""

    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label + 26


class EMNISTMappedDataset(Dataset):
    """EMNIST Letters 1-26 con etiquetas 0-25."""

    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.EMNIST(
            root, split="letters", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = torch.transpose(img, 1, 2)
        return img, label - 1


# ----------------------------------------------------------------
#  Augmentation
# ----------------------------------------------------------------

# Augmentation agresiva para simular condiciones de cámara real:
# motion blur, perspectiva, ruido, oclusión parcial.
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.12, 0.12),
        shear=10,
        scale=(0.82, 1.18),
    ),
    transforms.RandomPerspective(distortion_scale=0.35, p=0.5),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.8))], p=0.4
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


# ----------------------------------------------------------------
#  Entrenamiento
# ----------------------------------------------------------------

def entrenar():
    print(f"Dispositivo: {device}")
    (ROOT / "models").mkdir(exist_ok=True)

    print("Cargando EMNIST Letters…")
    emnist_train = EMNISTMappedDataset("./data", train=True,  transform=transform_train)
    emnist_test  = EMNISTMappedDataset("./data", train=False, transform=transform_test)

    print("Cargando MNIST Dígitos…")
    mnist_train = MNISTOffsetDataset("./data", train=True,  transform=transform_train)
    mnist_test  = MNISTOffsetDataset("./data", train=False, transform=transform_test)

    train_ds = ConcatDataset([emnist_train, mnist_train])
    test_ds  = ConcatDataset([emnist_test,  mnist_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    modelo = crear_modelo_cnn().to(device)

    # Label smoothing reduce confianza excesiva y mejora generalización
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW + OneCycleLR: convergencia más rápida y generalización superior a Adam+ReduceLR
    optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    mejor_acc = 0.0
    best_wts = copy.deepcopy(modelo.state_dict())

    print(f"\n[INFO] Iniciando entrenamiento ({EPOCHS} épocas)…\n")
    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────
        modelo.train()
        running_loss = corrects = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = modelo(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # OneCycleLR: paso por batch

            running_loss += loss.item() * inputs.size(0)
            corrects += (out.argmax(1) == labels).sum().item()
            total += inputs.size(0)

        train_acc  = corrects / total
        train_loss = running_loss / total

        # ── Validación ──────────────────────────────────────────
        modelo.eval()
        val_corrects = val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_corrects += (modelo(inputs).argmax(1) == labels).sum().item()
                val_total    += inputs.size(0)

        val_acc = val_corrects / val_total
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
            f"Loss {train_loss:.4f} | LR {lr_now:.2e}"
        )

        if val_acc > mejor_acc:
            mejor_acc = val_acc
            best_wts  = copy.deepcopy(modelo.state_dict())
            torch.save(best_wts, RUTA_MODELO)
            print(f"   ✓ Mejor modelo guardado  (val_acc={mejor_acc:.4f})")

    modelo.load_state_dict(best_wts)
    print(f"\n[OK] Entrenamiento completo. Mejor val_acc: {mejor_acc:.4f}")
    print(f"[OK] Modelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar()
