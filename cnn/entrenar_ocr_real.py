"""
cnn/entrenar_ocr_real.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entrena el clasificador de caracteres de placa usando SOLO datos REALES
(caracteres ya segmentados en Dataset_OCR_Placas/).

Por qué clasificador de caracteres (y no CRNN):
  El dataset son glifos reales segmentados (36 clases 0-9 A-Z). Los glifos
  son universales, así que sirven para un clasificador genérico. La secuencia
  de las placas del dataset es india (KA031351…), por lo que entrenar una CRNN
  end-to-end reaprendería ese formato y reintroduciría el dominio sintético que
  ya falló. Aquí solo aprendemos a reconocer cada carácter; el formato
  ecuatoriano ABC-NNNN se valida después, en la inferencia.

Mapeo de etiquetas:
  ImageFolder ordena las clases alfabéticamente -> ['0'..'9','A'..'Z'].
  Ese orden se GUARDA dentro del checkpoint ("classes") para que la inferencia
  use exactamente el mismo índice→carácter, sin depender del orden de
  modelo.CLASES.

Salida:
  models/ocr_char.pt  ->  {"state_dict": ..., "classes": [...], "img_size": 32}

Ejecutar desde la raíz del repo:
    .venv/bin/python cnn/entrenar_ocr_real.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from modelo import crear_modelo_cnn  # noqa: E402

# ----------------------------------------------------------------
#  Configuración
# ----------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

DATASET     = ROOT / "data" / "datasets" / "Dataset_OCR_Placas"
RUTA_MODELO = ROOT / "models" / "ocr_char.pt"
IMG_SIZE    = 48
EPOCHS      = 70
BATCH_SIZE  = 256
NUM_WORKERS = 8

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usar_amp = device.type == "cuda"


# ----------------------------------------------------------------
#  Transforms
# ----------------------------------------------------------------
def construir_transforms():
    # Aumentación Robusta para caracteres reales de placas
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Rotación aleatoria moderada para caracteres inclinados
        transforms.RandomRotation(8),
        # Traslación, escalado y deformación angular (shear)
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5),
        # Distorsión de perspectiva para simular ángulos de cámara
        transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        # Variabilidad de brillo y contraste por iluminación y sombras
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # Desenfoque gaussiano para simular pérdida de definición/movimiento
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        # Erasing aleatorio para simular tornillos, suciedad o daños en la placa
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    return tf_train, tf_eval


# ----------------------------------------------------------------
#  Evaluación
# ----------------------------------------------------------------
@torch.no_grad()
def evaluar(modelo, loader, criterion):
    modelo.eval()
    loss_acc = corr = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=usar_amp):
            out = modelo(x)
            loss = criterion(out, y)
        loss_acc += loss.item() * x.size(0)
        corr     += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return corr / total, loss_acc / total


@torch.no_grad()
def matriz_confusion_ambiguos(modelo, loader, classes):
    """Reporta los pares de caracteres más confundidos (O/0, I/1, B/8…)."""
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


# ----------------------------------------------------------------
#  Main
# ----------------------------------------------------------------
def entrenar():
    print(f"\nDispositivo: {device}  | AMP: {usar_amp}")
    print(f"Dataset: {DATASET}")
    if not DATASET.exists():
        print(f"[ERROR] No existe el dataset: {DATASET}")
        return

    tf_train, tf_eval = construir_transforms()
    ds_train = datasets.ImageFolder(str(DATASET / "train"), transform=tf_train)
    ds_val   = datasets.ImageFolder(str(DATASET / "valid"), transform=tf_eval)
    ds_test  = datasets.ImageFolder(str(DATASET / "test"),  transform=tf_eval)

    classes = ds_train.classes  # ['0'..'9','A'..'Z']
    print(f"Clases ({len(classes)}): {''.join(classes)}")
    print(f"Train {len(ds_train)} | Val {len(ds_val)} | Test {len(ds_test)}")
    assert ds_val.classes == classes == ds_test.classes, "Orden de clases inconsistente entre splits"

    pin = device.type == "cuda"
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(ds_val,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0)
    test_loader  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    modelo    = crear_modelo_cnn().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler(enabled=usar_amp)

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    mejor_acc = 0.0

    print("\n[INFO] Iniciando entrenamiento...\n")
    for epoch in range(1, EPOCHS + 1):
        modelo.train()
        run_loss = corr = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=usar_amp):
                out  = modelo(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item() * x.size(0)
            corr     += (out.argmax(1) == y).sum().item()
            total    += x.size(0)

        train_acc = corr / total
        val_acc, val_loss = evaluar(modelo, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train {train_acc:.4f} | "
              f"Val {val_acc:.4f} (loss {val_loss:.4f}) | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > mejor_acc:
            mejor_acc = val_acc
            torch.save({"state_dict": modelo.state_dict(),
                        "classes": classes,
                        "img_size": IMG_SIZE}, RUTA_MODELO)
            print(f"   ✓ Mejor modelo guardado (val_acc={mejor_acc:.4f})")

    # ── Evaluación final sobre TEST con el mejor modelo ──
    ckpt = torch.load(RUTA_MODELO, map_location=device, weights_only=False)
    modelo.load_state_dict(ckpt["state_dict"])
    test_acc, _ = evaluar(modelo, test_loader, criterion)
    print(f"\n[OK] Mejor val_acc: {mejor_acc:.4f}")
    print(f"[OK] TEST char acc: {test_acc:.4f}")
    print(f"[OK] Modelo: {RUTA_MODELO}")
    matriz_confusion_ambiguos(modelo, test_loader, classes)


if __name__ == "__main__":
    entrenar()
