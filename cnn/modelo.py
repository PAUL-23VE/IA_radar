"""
cnn/modelo.py
Arquitectura CNN con bloques residuales para reconocimiento de caracteres de placas.
Clases: A-Z (26) + 0-9 (10) = 36 clases en total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASES = len(CLASES)
TAMANO_IMAGEN = (32, 32)


import torchvision.models as models

class CNNPlacas(nn.Module):
    """
    ResNet18-based CNN para clasificación de caracteres de placa (48x48 o 64x64 -> 36 clases).
    ~11.1M parámetros. Alta precisión y latencia baja.
    """
    def __init__(self):
        super().__init__()
        
        # Cargar ResNet18 sin pesos pre-entrenados
        self.backbone = models.resnet18(weights=None)
        
        # Modificar la primera capa convolucional para que acepte 1 canal (escala de grises)
        # en lugar de 3 (RGB), y optimizarla para imágenes pequeñas (kernel 3, stride 1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remover el primer maxpool para mantener resolución
        
        # Modificar la capa final (fully connected) para que de 36 salidas
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, NUM_CLASES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def crear_modelo_cnn() -> CNNPlacas:
    return CNNPlacas()


def indice_a_caracter(indice: int) -> str:
    return CLASES[indice]


def caracter_a_indice(char: str) -> int:
    return CLASES.index(char.upper())


if __name__ == "__main__":
    m = crear_modelo_cnn()
    total = sum(p.numel() for p in m.parameters())
    print(f"Parámetros: {total:,}")
    out = m(torch.randn(8, 1, 32, 32))
    print(f"Salida shape: {out.shape}")
