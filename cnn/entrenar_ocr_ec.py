"""
cnn/entrenar_ocr_ec.py
Reentrena el clasificador de caracteres usando ResNet18 mezclando:
  - glifos REALES (Dataset_OCR_Placas/train)
  - glifos SINTÉTICOS TTF (dataset_propio_ttf) — tipografía EC (DejaVu Bold)

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
RUTA_MODELO   = ROOT / "models" / "ocr_char.pt" # Guardamos como el principal
IMG_SIZE   = 64
EPOCHS     = 60
BATCH_SIZE = 256
NUM_WORKERS = 4

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usar_amp = device.type == "cuda"

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # orden ImageFolder

def transforms_train():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Rotación aleatoria moderada
        transforms.RandomRotation(8),
        # Traslación, escalado y deformación angular
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5),
        # Distorsión de perspectiva para ángulos
        transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        # Variabilidad de brillo y contraste
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # Desenfoque gaussiano
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        # Erasing aleatorio para simular tornillos/suciedad
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
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

@torch.no_grad()
def matriz_confusion_ambiguos(modelo, loader, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    modelo.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=usar_amp):
            pred = modelo(x).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1

    print("\n[Confusión] Pares más confundidos (real → predicho):")
    errores = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                errores.append((cm[i, j], classes[i], classes[j]))
    errores.sort(reverse=True)
    for cuenta, real, pred in errores[:15]:
        print(f"   {real} → {pred}: {cuenta}")
    if not errores:
        print("   (sin errores en test)")

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
    ds_test  = cargar_imagefolder(DATASET_REAL / "test", tf_ev)

    print(f"Train total: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")
    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True,
                              persistent_workers=False)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=False)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    modelo = crear_modelo_cnn().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader),
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
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train {corr/total:.4f} | Val {va:.4f} (loss {vl:.4f}) | LR {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if va > mejor:
            mejor = va
            torch.save({"state_dict": modelo.state_dict(),
                        "classes": CLASSES, "img_size": IMG_SIZE}, RUTA_MODELO)
            print(f"   ✓ guardado (val={mejor:.4f})", flush=True)
    print(f"\n[OK] Mejor val: {mejor:.4f} -> {RUTA_MODELO}")
    
    # ── Evaluación final sobre TEST con el mejor modelo ──
    ckpt = torch.load(RUTA_MODELO, map_location=device, weights_only=False)
    modelo.load_state_dict(ckpt["state_dict"])
    test_acc, _ = evaluar(modelo, test_loader, criterion)
    print(f"[OK] TEST char acc: {test_acc:.4f}")
    matriz_confusion_ambiguos(modelo, test_loader, CLASSES)

if __name__ == "__main__":
    entrenar()
