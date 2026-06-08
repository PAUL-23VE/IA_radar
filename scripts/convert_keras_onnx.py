"""
Convierte los modelos Keras (.keras HDF5) del amigo a ONNX, para correrlos con
onnxruntime en el venv de Python 3.14 (donde TensorFlow no tiene wheels).

Se ejecuta UNA sola vez dentro de un contenedor python:3.11 con tensorflow-cpu.
Salida: models/amigo_ocr_cnn.onnx y models/amigo_seg_unet.onnx
"""
import tensorflow as tf
import tf2onnx

BASE = "Plate_Detection_Segmentation_OCR_sin_dependencias/ml/models"
PARES = [
    (f"{BASE}/ocr/Modelos/best_cnn_ocr.keras", "models/amigo_ocr_cnn.onnx"),
    (f"{BASE}/char_segmentation/Models/best_char_segmentation_unet.keras",
     "models/amigo_seg_unet.onnx"),
]

for src, dst in PARES:
    m = tf.keras.models.load_model(src, compile=False)
    print(f"\n{src}\n  input_shape={m.input_shape} output_shape={m.output_shape}")
    spec = (tf.TensorSpec(m.input_shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=dst)
    print(f"  -> {dst}  OK")
