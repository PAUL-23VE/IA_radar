"""
cnn/entrenar_ocr_ec.py
Reentrena el clasificador de caracteres mezclando:
  - glifos REALES (Dataset_OCR_Placas/train)        — variedad de captura real
  - glifos SINTÉTICOS TTF (dataset_propio_ttf)       — tipografía EC (DejaVu Bold)

Motivo: el modelo entrenado SOLO con Dataset_OCR_Placas tenía gap de dominio
(89.7% en su test, pero ~2/7 en placas EC reales: G→H, T→3). La fuente del
dataset no coincide con la placa ecuatoriana. Añadir glifos sintéticos en la
fuente real de placa cierra ese gap.

Salida: models/ocr_char_ec.pt  {state_dict, classes, img_size}

Ejecutar desde la raíz:
    .venv/bin/python cnn/entrenar_ocr_ec.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from modelo import crear_modelo_cnn  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

DATASET_REAL  = ROOT / "data" / "datasets" / "Dataset_OCR_Placas"
DATASET_SYNTH = ROOT / "data" / "datasets" / "dataset_propio_ttf"
RUTA_MODELO   = ROOT / "models" / "ocr_char_ec.pt"
IMG_SIZE   = 48
EPOCHS     = 25
BATCH_SIZE = 256
NUM_WORKERS = 8

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usar_amp = device.type == "cuda"

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # orden ImageFolder


def transforms_train():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(6),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
    ])


def transforms_eval():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def cargar_imagefolder(raiz: Path, tf):
    """Carga ImageFolder forzando el orden de clases CLASSES."""
    ds = datasets.ImageFolder(str(raiz), transform=tf)
    # Verificar que las clases coincidan con el orden canónico
    assert ds.classes == CLASSES, f"Clases inesperadas en {raiz}: {ds.classes}"
    return ds


@torch.no_grad()
def evaluar(modelo, loader, criterion):
    modelo.eval()
    loss_acc = corr = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=usar_amp):
            out = modelo(x); loss = criterion(out, y)
        loss_acc += loss.item() * x.size(0)
        corr += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return corr / total, loss_acc / total


def entrenar():
    print(f"Dispositivo: {device} | AMP: {usar_amp}")
    tf_tr, tf_ev = transforms_train(), transforms_eval()

    partes = [cargar_imagefolder(DATASET_REAL / "train", tf_tr)]
    if DATASET_SYNTH.exists():
        partes.append(cargar_imagefolder(DATASET_SYNTH, tf_tr))
        print(f"Sintético TTF: {len(partes[-1])} imgs")
    else:
        print("[WARN] No existe dataset_propio_ttf; entrenando solo con reales.")
    ds_train = ConcatDataset(partes)
    ds_val   = cargar_imagefolder(DATASET_REAL / "valid", tf_ev)

    print(f"Train total: {len(ds_train)} | Val: {len(ds_val)}")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)

    modelo = crear_modelo_cnn().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=usar_amp)

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    mejor = 0.0
    for epoch in range(1, EPOCHS + 1):
        modelo.train()
        rl = corr = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=usar_amp):
                out = modelo(x); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            rl += loss.item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item(); total += x.size(0)
        va, vl = evaluar(modelo, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train {corr/total:.4f} | Val {va:.4f} (loss {vl:.4f})")
        if va > mejor:
            mejor = va
            torch.save({"state_dict": modelo.state_dict(),
                        "classes": CLASSES, "img_size": IMG_SIZE}, RUTA_MODELO)
            print(f"   ✓ guardado (val={mejor:.4f})")
    print(f"\n[OK] Mejor val: {mejor:.4f} -> {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar()
