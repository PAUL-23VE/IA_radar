"""
cnn/modelo.py
Definición de la arquitectura CNN para reconocer caracteres de placas.
Clases: A-Z (26) + 0-9 (10) = 36 clases en total.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# 36 caracteres posibles en una placa ecuatoriana
CLASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NUM_CLASES = len(CLASES)          # 36
TAMANO_IMAGEN = (32, 32)          # cada char reescalado a 32×32 px


def crear_modelo_cnn() -> tf.keras.Model:
    """
    CNN de 3 bloques convolucionales + clasificador.

    Entrada:  (32, 32, 1)  — imagen en escala de grises
    Salida:   (36,)        — probabilidades por clase (softmax)
    """

    modelo = models.Sequential(name="CNN_Placas", layers=[

        # ── Bloque 1 ─────────────────────────────────────────
        layers.Conv2D(32, (3,3), activation='relu',
                      padding='same', input_shape=(32, 32, 1),
                      name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 32×32 → 16×16
        layers.Dropout(0.25),

        # ── Bloque 2 ─────────────────────────────────────────
        layers.Conv2D(64, (3,3), activation='relu',
                      padding='same', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 16×16 → 8×8
        layers.Dropout(0.25),

        # ── Bloque 3 ─────────────────────────────────────────
        layers.Conv2D(128, (3,3), activation='relu',
                      padding='same', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),          # 8×8 → 4×4
        layers.Dropout(0.25),

        # ── Clasificador ─────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(NUM_CLASES, activation='softmax', name='salida')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


def indice_a_caracter(indice: int) -> str:
    """Convierte el índice predicho al carácter correspondiente."""
    return CLASES[indice]


def caracter_a_indice(char: str) -> int:
    """Convierte un carácter a su índice en la lista de clases."""
    return CLASES.index(char.upper())


if __name__ == "__main__":
    m = crear_modelo_cnn()
    m.summary()
