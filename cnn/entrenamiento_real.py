"""
cnn/entrenamiento_real.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reentrena la CNN usando una mezcla de:
  1. dataset_propio/  → recortes reales de placas ecuatorianas
  2. EMNIST + MNIST   → complemento para las clases poco representadas

El modelo resultante se guarda en:
  models/modelo_entrenado.pt   (sobreescribe el anterior)

Ejecutar desde la raíz del repo:
    .venv/bin/python cnn/entrenamiento_real.py
"""

import os
import sys
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from modelo import crear_modelo_cnn, CLASES, NUM_CLASES

# ----------------------------------------------------------------
#  Configuración
# ----------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

RUTA_MODELO      = str(ROOT / "models" / "modelo_entrenado.pt")
DATASET_REAL     = ROOT / "dataset_propio"
EPOCHS           = 20
BATCH_SIZE       = 128
# Porcentaje de imágenes EMNIST/MNIST a mezclar (0.0 = solo reales, 1.0 = 50/50)
MEZCLA_SINTETICA = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------
#  Dataset real (dataset_propio/)
# ----------------------------------------------------------------
class PlacasRealDataset(Dataset):
    """
    Lee la carpeta dataset_propio/ organizada como ImageFolder:
        dataset_propio/
            A/ img1.png  img2.png ...
            B/ ...
            0/ ...
    Convierte cada imagen a escala de grises 32x32 y la etiqueta
    con el índice correspondiente en CLASES.
    """
    def __init__(self, root: Path, transform=None):
        self.samples = []
        self.transform = transform
        for clase in CLASES:
            carpeta = root / clase
            if not carpeta.exists():
                continue
            idx = CLASES.index(clase)
            for ruta in sorted(carpeta.glob("*.png")):
                self.samples.append((str(ruta), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ruta, label = self.samples[i]
        import cv2
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, 32), dtype=np.uint8)
        img = img.astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0)   # (1, 32, 32)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


# ----------------------------------------------------------------
#  Datasets sintéticos complementarios
# ----------------------------------------------------------------
class MNISTOffsetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transform)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label + 26   # 0-9 → 26-35


class EMNISTMappedDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.EMNIST(root, split='letters', train=train,
                                  download=True, transform=transform)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = torch.transpose(img, 1, 2)
        return img, label - 1   # 1-26 → 0-25 (A-Z)


# ----------------------------------------------------------------
#  Main: entrenamiento mixto
# ----------------------------------------------------------------
def entrenar():
    print(f"\nDispositivo: {device}")
    print(f"Dataset real: {DATASET_REAL}")

    # ── Augmentation para imágenes reales ───────────────────────
    aug_real = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])
    aug_sintetico = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # ── Dataset real ────────────────────────────────────────────
    ds_real_train = PlacasRealDataset(DATASET_REAL, transform=aug_real)
    n_real = len(ds_real_train)
    print(f"Ejemplos reales encontrados: {n_real}")

    if n_real == 0:
        print("\n[ERROR] No se encontraron ejemplos reales.")
        print("Ejecuta primero:  .venv/bin/python scripts/extraer_caracteres.py")
        return

    # Distribución por clase (para log)
    conteo_clases = [0] * NUM_CLASES
    for _, lbl in ds_real_train.samples:
        conteo_clases[lbl] += 1
    print("\nDistribución del dataset real:")
    for i, c in enumerate(CLASES):
        if conteo_clases[i] > 0:
            print(f"  {c}: {conteo_clases[i]}")

    # ── Dataset sintético complementario ────────────────────────
    data_dir = str(ROOT / "data")
    emnist_train = EMNISTMappedDataset(data_dir, train=True,  transform=aug_sintetico)
    emnist_test  = EMNISTMappedDataset(data_dir, train=False, transform=test_tf)
    mnist_train  = MNISTOffsetDataset(data_dir,  train=True,  transform=aug_sintetico)
    mnist_test   = MNISTOffsetDataset(data_dir,  train=False, transform=test_tf)

    # ── Combinar: usar sólo una fracción de sintéticos ──────────
    n_sintetico = int(n_real * MEZCLA_SINTETICA / (1 - MEZCLA_SINTETICA))
    n_sintetico = min(n_sintetico, len(emnist_train) + len(mnist_train))
    print(f"\nEjemplos sintéticos a mezclar: {n_sintetico}")

    # Submuestrear sintéticos aleatoriamente
    indices_sin = torch.randperm(len(emnist_train) + len(mnist_train))[:n_sintetico]
    sintetico_full = ConcatDataset([emnist_train, mnist_train])
    from torch.utils.data import Subset
    sintetico_sub = Subset(sintetico_full, indices_sin.tolist())

    train_dataset = ConcatDataset([ds_real_train, sintetico_sub])
    test_dataset  = ConcatDataset([
        EMNISTMappedDataset(data_dir, train=False, transform=test_tf),
        MNISTOffsetDataset(data_dir,  train=False, transform=test_tf),
    ])

    print(f"Total train: {len(train_dataset)} | test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Cargar modelo base y afinar ─────────────────────────────
    modelo = crear_modelo_cnn().to(device)
    if Path(RUTA_MODELO).exists():
        print(f"\n[INFO] Cargando pesos previos de {RUTA_MODELO} (fine-tuning)")
        modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=device, weights_only=True))
    else:
        print("\n[INFO] Entrenando desde cero (no se encontró modelo previo)")

    criterion = nn.CrossEntropyLoss()
    # LR más bajo para fine-tuning
    optimizer = optim.Adam(modelo.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=3)

    mejor_acc = 0.0
    best_wts  = copy.deepcopy(modelo.state_dict())

    print("\n[INFO] Iniciando entrenamiento...\n")
    for epoch in range(1, EPOCHS + 1):
        # Fase train
        modelo.train()
        running_loss = corrects = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = modelo(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(out, 1)
            corrects += (preds == labels).sum().item()
            total    += inputs.size(0)

        train_acc  = corrects / total
        train_loss = running_loss / total

        # Fase validación
        modelo.eval()
        val_loss = val_corrects = val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out  = modelo(inputs)
                loss = criterion(out, labels)
                val_loss     += loss.item() * inputs.size(0)
                _, preds = torch.max(out, 1)
                val_corrects += (preds == labels).sum().item()
                val_total    += inputs.size(0)

        val_acc  = val_corrects / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_acc)

        if val_acc > mejor_acc:
            mejor_acc = val_acc
            best_wts  = copy.deepcopy(modelo.state_dict())
            torch.save(modelo.state_dict(), RUTA_MODELO)
            print(f"   ✓ Nuevo mejor modelo guardado  (val_acc={mejor_acc:.4f})")

    print(f"\n[OK] Entrenamiento completado. Mejor val_acc: {mejor_acc:.4f}")
    print(f"[OK] Modelo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar()
