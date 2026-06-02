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


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class CNNPlacas(nn.Module):
    """
    ResNet-style CNN para clasificación de caracteres de placa (32×32 → 36 clases).
    ~1.2M parámetros. Latencia ~1ms/batch(8) en GPU.
    """

    def __init__(self):
        super().__init__()

        # 1×32×32 → 64×16×16
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.res1 = ResBlock(64)

        # 64×16×16 → 128×8×8
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.res2 = ResBlock(128)

        # 128×8×8 → 256×4×4
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
        )
        self.res3 = ResBlock(256)

        # 256×4×4 → 36
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        return self.head(x)


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
