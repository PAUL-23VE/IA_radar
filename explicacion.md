# Arquitectura del Sistema de Reconocimiento de Placas

Este documento explica detalladamente el funcionamiento del pipeline de Inteligencia Artificial desarrollado para el reconocimiento de placas vehiculares ecuatorianas. El sistema fue diseñado desde cero para evitar depender exclusivamente de herramientas preestablecidas, implementando una red neuronal convolucional (CNN) propia.

## Pipeline General

El sistema opera en tres fases secuenciales:

1. **Detección de la Placa (YOLOv11):** Localiza la placa dentro de la imagen del vehículo.
2. **Segmentación de Caracteres (OpenCV):** Aísla cada letra y número de la placa.
3. **Clasificación (CNN PyTorch):** Reconoce qué carácter es cada recorte.

---

## 1. Detección (YOLO)
Se utiliza un modelo YOLOv11 entrenado específicamente para detectar la caja delimitadora (bounding box) de las placas vehiculares ecuatorianas. Una vez detectada la placa, se recorta esa región de la imagen original para enviarla a la siguiente fase.

## 2. Segmentación Multi-estrategia (OpenCV)
Las placas ecuatorianas presentan desafíos como la palabra "ECUADOR" en la parte superior, tornillos, reflejos metálicos y suciedad. Para resolver esto, el segmentador (`cnn/inferencia.py`) implementa:

* **Recorte de zona superior:** Se elimina automáticamente el 18% superior del recorte para ignorar el texto "ECUADOR" que suele confundir al reconocedor.
* **Escalado Dinámico:** Si la detección es muy pequeña (resolución pobre), se interpola y escala la imagen a un mínimo de 120px de ancho.
* **Grid de Binarización (50 combinaciones):** En lugar de usar un solo umbral, el algoritmo prueba 5 umbrales distintos (Adaptativo, Otsu, Fijo) × 2 modos (normal e invertido) × 5 tamaños de recorte inferior.
* **Proyección Vertical (Fallback Matemático):** Si las letras están muy juntas, se suma el valor de los píxeles por columna (histograma de proyección) para encontrar los "valles" de separación entre caracteres.

El segmentador busca aislar exactamente entre 6 y 7 componentes conexos (las letras y números).

## 3. Red Neuronal Convolucional (PyTorch)
En lugar de usar EasyOCR como motor principal, se construyó la red `CNNPlacas` desde cero (`cnn/modelo.py`).

### Arquitectura de la CNN
* **Entrada:** Imágenes en escala de grises de 32x32 píxeles.
* **Capas Convolucionales:** Tres bloques convolucionales (`Conv2d` -> `BatchNorm` -> `ReLU` -> `MaxPool2d`). Estas capas actúan como "extractores de características", aprendiendo a detectar bordes, curvas y formas de las letras.
* **Capas Densas (Fully Connected):** Dos capas lineales que aplican `Dropout` para evitar el sobreajuste (overfitting) y mapean las características extraídas a una de las 36 clases posibles (A-Z, 0-9).

### Entrenamiento Híbrido (Fine-Tuning)
El modelo no fue entrenado genéricamente. Para lograr una precisión superior al 94%, se implementó un flujo de *Transfer Learning*:
1. **Fase Inicial:** El modelo aprendió las formas básicas de letras y números usando los datasets públicos EMNIST y MNIST.
2. **Extracción Real:** Se desarrolló un script (`scripts/extraer_caracteres.py`) que extrajo automáticamente 1,676 recortes individuales de caracteres directamente de las fotos de placas ecuatorianas del dataset.
3. **Fine-Tuning:** Se reentrenó la CNN mezclando los caracteres sintéticos (EMNIST) con los 1,676 recortes metálicos reales. Esto enseñó a la red a tolerar brillos, óxido, y la tipografía exacta de la Agencia Nacional de Tránsito.

### Predicción Posicional
Un aporte algorítmico clave fue la "restricción por logits". Sabemos que las placas ecuatorianas tienen un formato fijo (3 letras seguidas de 3-4 números). En la inferencia, la red enmascara (multiplica por cero) la probabilidad de predecir números en las primeras 3 posiciones, y enmascara las letras en las posiciones finales, garantizando resultados como `ABC-1234`.

---

## 4. Validación y Red de Seguridad
Finalmente, el sistema valida la cadena de texto contra la expresión regular `^[A-Z]{3}-\d{2,4}$`. 

**Fallback:** Para el ~15% de los casos en que la placa está cubierta de barro o extremadamente borrosa (impidiendo la separación física por OpenCV), el sistema invoca una red neuronal recurrente LSTM de rescate (EasyOCR) como plan B, asegurando robustez en condiciones extremas y permitiendo al sistema cruzar la meta del 86.7% de precisión final.
