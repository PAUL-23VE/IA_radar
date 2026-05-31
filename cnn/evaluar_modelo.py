import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

# Asegurar que se puede importar modelo
sys.path.insert(0, os.path.dirname(__file__))
from modelo import NUM_CLASES, CLASES

def evaluar_modelo():
    ruta_modelo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "modelo_entrenado.h5")
    if not os.path.exists(ruta_modelo):
        print(f"[Error] El modelo no existe en {ruta_modelo}. Entrénalo primero.")
        return

    print("[INFO] Cargando modelo...")
    modelo = tf.keras.models.load_model(ruta_modelo)

    print("[INFO] Cargando datasets de prueba...")
    BATCH_SIZE = 128

    # Cargar EMNIST Test
    ds_emnist_test = tfds.load('emnist/letters', split='test', as_supervised=True)
    def prep_emnist(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [32, 32])
        label = tf.cast(label - 1, tf.int64)
        return img, label
    ds_emnist = ds_emnist_test.map(prep_emnist)

    # Cargar MNIST Test
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test[..., np.newaxis].astype('float32') / 255.0
    x_test = tf.image.resize(x_test, [32, 32]).numpy()
    y_test = (y_test + 26).astype(np.int64)
    ds_mnist = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Combinar Test Set
    test_dataset = ds_emnist.concatenate(ds_mnist).batch(BATCH_SIZE)

    print("[INFO] Realizando inferencia sobre todo el set de pruebas...")
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = modelo.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n[INFO] Evaluacion del Modelo completada.")
    print("="*60)
    print(" REPORTE DE CLASIFICACIÓN ")
    print("="*60)
    reporte = classification_report(y_true, y_pred, target_names=list(CLASES))
    print(reporte)

    # Generar Matriz de Confusión
    print("[INFO] Generando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g',
                xticklabels=list(CLASES), yticklabels=list(CLASES))
    plt.title('Matriz de Confusión - Red Neuronal Placas')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción de la CNN')
    
    # Guardar en raíz
    ruta_guardado = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'matriz_confusion.png')
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300)
    print(f"[OK] Matriz de Confusión guardada exitosamente en '{ruta_guardado}'")

if __name__ == "__main__":
    evaluar_modelo()
