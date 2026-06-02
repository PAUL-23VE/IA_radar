"""
cnn/entrenamiento.py
Entrena la CNN usando EMNIST Letters (A-Z) + MNIST (0-9) en PyTorch.
Guarda el modelo entrenado en models/modelo_entrenado.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import os
import copy
from pathlib import Path

# Fix relative imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from modelo import crear_modelo_cnn, NUM_CLASES

# ----------------------------------------------------------------
#  Reproducibilidad y Config
# ----------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
RUTA_MODELO = str(ROOT / "models" / "modelo_entrenado.pt")
EPOCHS      = 15 # Reduced for speed, PyTorch usually converges fast
BATCH_SIZE  = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
#  1. DATASETS PERSONALIZADOS
# ----------------------------------------------------------------

class MNISTOffsetDataset(Dataset):
    """Carga MNIST (0-9) y suma 26 a las etiquetas para que sean 26-35."""
    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.MNIST(root, train=train, download=True, transform=transform)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return img, label + 26

class EMNISTMappedDataset(Dataset):
    """Carga EMNIST letters (1-26) y resta 1 a las etiquetas para que sean 0-25."""
    def __init__(self, root, train=True, transform=None):
        self.ds = datasets.EMNIST(root, split='letters', train=train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # EMNIST letters are transposed by default, transpose it back
        img = torch.transpose(img, 1, 2)
        return img, label - 1

def entrenar():
    print(f"Dispositivo de entrenamiento: {device}")
    
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    print("Cargando EMNIST Letters...")
    emnist_train = EMNISTMappedDataset('./data', train=True, transform=transform_train)
    emnist_test = EMNISTMappedDataset('./data', train=False, transform=transform_test)

    print("Cargando MNIST Digitos...")
    mnist_train = MNISTOffsetDataset('./data', train=True, transform=transform_train)
    mnist_test = MNISTOffsetDataset('./data', train=False, transform=transform_test)

    train_dataset = ConcatDataset([emnist_train, mnist_train])
    test_dataset = ConcatDataset([emnist_test, mnist_test])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    modelo = crear_modelo_cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    mejor_acc = 0.0
    best_model_wts = copy.deepcopy(modelo.state_dict())

    print("\n[INFO] Iniciando entrenamiento...")
    for epoch in range(EPOCHS):
        # Entrenar
        modelo.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
        train_loss = running_loss / total
        train_acc = corrects.double() / total
        
        # Validacion
        modelo.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = modelo(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > mejor_acc:
            mejor_acc = val_acc
            best_model_wts = copy.deepcopy(modelo.state_dict())
            torch.save(modelo.state_dict(), RUTA_MODELO)
            print(f"   [!] Mejor modelo guardado con exactitud: {mejor_acc:.4f}")

    print(f"\n[OK] Entrenamiento completado. Mejor exactitud de validacion: {mejor_acc:.4f}")
    print(f"[OK] Modelo guardado en: {RUTA_MODELO}")

if __name__ == "__main__":
    entrenar()
