"""
cnn/modelo.py
Definición de la arquitectura CNN en PyTorch para reconocer caracteres de placas.
Clases: A-Z (26) + 0-9 (10) = 36 clases en total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASES = len(CLASES)          # 36
TAMANO_IMAGEN = (32, 32)          # cada char reescalado a 32x32 px

class CNNPlacas(nn.Module):
    def __init__(self):
        super(CNNPlacas, self).__init__()
        
        # Bloque 1: Entrada 1x32x32 -> Salida 32x16x16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Bloque 2: Entrada 32x16x16 -> Salida 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Bloque 3: Entrada 64x8x8 -> Salida 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Clasificador
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, NUM_CLASES)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        
        x = x.view(-1, 128 * 4 * 4)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop4(x)
        
        x = self.fc2(x)
        return x

def crear_modelo_cnn():
    return CNNPlacas()

def indice_a_caracter(indice: int) -> str:
    """Convierte el índice predicho al carácter correspondiente."""
    return CLASES[indice]

def caracter_a_indice(char: str) -> int:
    """Convierte un carácter a su índice en la lista de clases."""
    return CLASES.index(char.upper())

if __name__ == "__main__":
    m = crear_modelo_cnn()
    print(m)
    test_input = torch.randn(1, 1, 32, 32)
    output = m(test_input)
    print("Salida shape:", output.shape)
