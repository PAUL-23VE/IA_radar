"""
cnn/entrenamiento.py
Entrena la CNN usando EMNIST Letters (A-Z) + MNIST (0-9).
Guarda el modelo entrenado en cnn/modelo_entrenado.h5

Ejecutar: python cnn/entrenamiento.py
Tiempo estimado: 10-20 min (CPU) / 3-5 min (GPU)
"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                         EarlyStopping,
                                         ReduceLROnPlateau)
from modelo import crear_modelo_cnn, NUM_CLASES

# ----------------------------------------------------------------
#  Reproducibilidad
# ----------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

RUTA_MODELO = "models/modelo_entrenado.h5"
EPOCHS      = 30
BATCH_SIZE  = 128


# ----------------------------------------------------------------
#  1. CARGAR Y PREPARAR DATOS
# ----------------------------------------------------------------

def cargar_emnist_letras():
    """
    EMNIST Letters: 26 clases (A=0 … Z=25), imágenes 28×28.
    Las etiquetas originales van de 1-26, las ajustamos a 0-25.
    """
    print("Cargando EMNIST Letters...")
    (ds_train, ds_test), info = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    def preparar(imagen, etiqueta):
        imagen = tf.cast(imagen, tf.float32) / 255.0
        imagen = tf.image.resize(imagen, [32, 32])
        etiqueta = tf.cast(etiqueta - 1, tf.int64)  # 1-26 → 0-25 (A-Z) y cast a int64
        return imagen, etiqueta

    train = ds_train.map(preparar).cache().shuffle(10000).batch(BATCH_SIZE)
    test  = ds_test.map(preparar).cache().batch(BATCH_SIZE)
    return train, test


def cargar_mnist_digitos():
    """
    MNIST: 10 clases (0-9), imágenes 28×28.
    Reetiquetamos: dígito 0 → clase 26, dígito 1 → 27, ... dígito 9 → 35
    para que no choquen con las letras (0-25).
    """
    print("Cargando MNIST dígitos...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[..., np.newaxis].astype('float32') / 255.0
    x_test  = x_test[..., np.newaxis].astype('float32')  / 255.0

    # Resize 28×28 → 32×32
    x_train = tf.image.resize(x_train, [32, 32]).numpy()
    x_test  = tf.image.resize(x_test,  [32, 32]).numpy()

    # Re-etiquetar: dígito d → clase (26 + d) y convertir a int64
    y_train = (y_train + 26).astype(np.int64)
    y_test  = (y_test  + 26).astype(np.int64)

    ds_train = (tf.data.Dataset
                .from_tensor_slices((x_train, y_train))
                .cache().shuffle(10000).batch(BATCH_SIZE))
    ds_test  = (tf.data.Dataset
                .from_tensor_slices((x_test, y_test))
                .cache().batch(BATCH_SIZE))
    return ds_train, ds_test


def augment_cv2(imagen):
    """Aplica Data Augmentation avanzado usando OpenCV."""
    # imagen is 32x32x1 float32 en [0, 1]
    img = (imagen * 255).astype(np.uint8)
    
    # 1. Ruido
    if np.random.rand() > 0.5:
        ruido = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
        
    # 2. Blur / Motion Blur
    prob_blur = np.random.rand()
    if prob_blur > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif prob_blur > 0.4:
        # Motion Blur
        kernel_size = 3
        kernel_mb = np.zeros((kernel_size, kernel_size))
        kernel_mb[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_mb = kernel_mb / kernel_size
        img = cv2.filter2D(img, -1, kernel_mb)
        
    # 3. Perspective Transform (Inclinación)
    if np.random.rand() > 0.5:
        rows, cols = img.shape[:2]
        pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        d = 2 # max shift
        pts2 = np.float32([
            [np.random.randint(0, d), np.random.randint(0, d)],
            [cols-np.random.randint(0, d), np.random.randint(0, d)],
            [np.random.randint(0, d), rows-np.random.randint(0, d)],
            [cols-np.random.randint(0, d), rows-np.random.randint(0, d)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
    # 4. Rotación
    if np.random.rand() > 0.5:
        angulo = np.random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angulo, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
    img = img.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        
    return img

def tf_augment(imagen, etiqueta):
    """Wrapper para aplicar augment_cv2 y variaciones de TF en tf.data."""
    # Data augmentation en numpy/cv2
    imagen_aug = tf.numpy_function(func=augment_cv2, inp=[imagen], Tout=tf.float32)
    imagen_aug.set_shape((32, 32, 1))
    
    # 5. Cambios de brillo nativos de TF
    imagen_aug = tf.image.random_brightness(imagen_aug, max_delta=0.2)
    imagen_aug = tf.clip_by_value(imagen_aug, 0.0, 1.0)
    
    return imagen_aug, etiqueta

def combinar_datasets(ds1_train, ds1_test, ds2_train, ds2_test):
    """Une los dos datasets, los mezcla y aplica data augmentation al train set."""
    train = ds1_train.unbatch().concatenate(ds2_train.unbatch())
    test  = ds1_test.unbatch().concatenate(ds2_test.unbatch())
    
    # Aplicar Data Augmentation SOLAMENTE al set de entrenamiento
    train = train.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    train = train.shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test  = test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train, test


# ----------------------------------------------------------------
#  2. ENTRENAMIENTO
# ----------------------------------------------------------------

def entrenar():
    # Datos
    emnist_train, emnist_test = cargar_emnist_letras()
    mnist_train,  mnist_test  = cargar_mnist_digitos()
    train, test = combinar_datasets(emnist_train, emnist_test,
                                     mnist_train,  mnist_test)

    # Modelo
    modelo = crear_modelo_cnn()
    modelo.summary()

    # Callbacks
    callbacks = [
        # Guardar el mejor modelo durante el entrenamiento
        ModelCheckpoint(
            filepath=RUTA_MODELO,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Detener si no mejora en 5 épocas
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir learning rate si se estanca
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Entrenar
    print("\n[INFO] Iniciando entrenamiento...")
    historia = modelo.fit(
        train,
        epochs=EPOCHS,
        validation_data=test,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluación final
    print("\n[INFO] Evaluación final en test set:")
    loss, acc = modelo.evaluate(test, verbose=0)
    print(f"   Loss:     {loss:.4f}")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"\n[OK] Modelo guardado en: {RUTA_MODELO}")

    return historia


if __name__ == "__main__":
    entrenar()
