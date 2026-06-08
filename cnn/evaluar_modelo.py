"""
cnn/evaluar_modelo.py
Evalúa la CNN (PyTorch) sobre EMNIST + MNIST de prueba.
Genera reporte de clasificación y matriz de confusión.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from modelo import CLASES, NUM_CLASES, crear_modelo_cnn
from entrenamiento import EMNISTMappedDataset, MNISTOffsetDataset, transform_test


def evaluar_modelo():
    ruta = ROOT / "models" / "modelo_entrenado.pt"
    if not ruta.exists():
        print(f"[Error] No existe {ruta}. Ejecuta primero: python cnn/entrenamiento.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo_cnn().to(device)
    modelo.load_state_dict(
        torch.load(str(ruta), map_location=device, weights_only=True)
    )
    modelo.eval()
    print(f"Modelo cargado desde {ruta}")
    print(f"Dispositivo: {device}")

    data_dir = str(ROOT / "data" / "datasets")
    test_ds = ConcatDataset([
        EMNISTMappedDataset(data_dir, train=False, transform=transform_test),
        MNISTOffsetDataset(data_dir,  train=False, transform=transform_test),
    ])
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Evaluando {len(test_ds):,} muestras…")
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = modelo(imgs).argmax(1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\nExactitud global: {acc:.4f} ({acc*100:.2f}%)")
    print("\nReporte de clasificación:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=list(CLASES)))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm, annot=False, cmap="Blues", fmt="g",
        xticklabels=list(CLASES), yticklabels=list(CLASES),
    )
    plt.title(f"Matriz de Confusión — CNN Placas (acc={acc:.3f})")
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    ruta_png = ROOT / "matriz_confusion.png"
    plt.tight_layout()
    plt.savefig(str(ruta_png), dpi=150)
    print(f"\nMatriz guardada en {ruta_png}")


if __name__ == "__main__":
    evaluar_modelo()
