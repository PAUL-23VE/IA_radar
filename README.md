# Sistema de Reconocimiento de Placas Vehiculares
**Materia:** Inteligencia Artificial  
**Universidad:** UTA

Detección de placas con **YOLOv11** (`best.pt`) + lectura con **EasyOCR**,
medición de velocidad por cruce de dos líneas y sanción por **lógica difusa**.

## Estructura del proyecto
```
radar/
├── best.pt                 # Pesos YOLOv11 — detector de placas (1 clase)
├── database/
│   ├── schema.sql          # Estructura de la BD PostgreSQL
│   ├── seed_data.sql       # Datos ficticios (50 vehículos)
│   └── db_connection.py    # Conexión y consultas a PostgreSQL
├── cnn/
│   └── inferencia.py       # YOLOv11 detecta placa → EasyOCR lee → valida (ABC-1234)
│   # (legacy) modelo.py / entrenamiento.py / evaluar_modelo.py:
│   #          CNN de caracteres EMNIST/MNIST, ya NO se usa en el pipeline activo
├── velocidad/
│   └── logica_difusa.py    # scikit-fuzzy: velocidad → sanción
├── utils/
│   └── camara.py           # Captura de frames desde iPhone
└── main.py                 # Pipeline completo integrado
```

## Flujo
1. Medir velocidad (cruce de Línea A → Línea B) y clasificar con lógica difusa.
2. `best.pt` (YOLOv11) detecta la región de la placa en el frame.
3. EasyOCR lee el texto del recorte; se valida al formato ecuatoriano `ABC-1234`.
4. Se consulta PostgreSQL y, si corresponde, se registra la multa.

## Instalación
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
La primera ejecución de EasyOCR descarga sus modelos de detección/reconocimiento.

## GPU / CPU
Por defecto el sistema usa GPU si PyTorch detecta CUDA; si no, cae a CPU.
```bash
# Automático: GPU si existe, CPU si no
python scripts/probar_dataset.py --split test --limit 30 --device auto

# Forzar tu NVIDIA/CUDA
python scripts/probar_dataset.py --split test --limit 30 --device cuda

# Forzar CPU para equipos sin tarjeta gráfica
python scripts/probar_dataset.py --split test --limit 30 --device cpu

# También sirve con variables de entorno, útil para main.py
RADAR_DEVICE=cuda RADAR_OCR_VARIANTS=2 python main.py laptop
```

## Cómo correr
```bash
# 1. Crear la base de datos
psql -U postgres -f database/schema.sql
psql -U postgres -d placas_db -f database/seed_data.sql

# 2. Probar el detector + OCR sobre una imagen
python cnn/inferencia.py ruta/a/imagen.jpg   # genera resultado.jpg

# 3. Probar muchas imágenes del dataset local
python scripts/probar_dataset.py --split test --limit 30 --device auto
python scripts/probar_dataset.py --split test --limit 30 --variants 2  # modo rápido
# resultados: dev_outputs/dataset_ocr/resultados.csv + imágenes anotadas

# Crear plantilla para anotar la placa real y medir exactitud real
python scripts/crear_ground_truth.py \
  --results dev_outputs/dataset_ocr/resultados.csv \
  --out dataset_combinado/ground_truth_test.csv
python scripts/probar_dataset.py --split test --limit 30 \
  --ground-truth dataset_combinado/ground_truth_test.csv

# 4. Probar cámara y grabar evidencia
python scripts/camara_dev.py --source 0 --device auto
python scripts/camara_dev.py --source 0 --device cuda --variants 5 --every 60  # más robusto en GPU
python scripts/camara_dev.py --source 0 --device cpu   # fallback para equipos sin GPU
# en la ventana: r = grabar/parar, s = guardar frame, espacio = OCR inmediato, q = salir

# 5. Ejecutar el sistema completo
python main.py demo     # solo BD + lógica difusa (sin cámara)
python main.py laptop   # cámara del portátil
python main.py live     # stream del teléfono (utils/camara.py)
```

## Reentrenar el detector (opcional)
`best.pt` ya viene entrenado (YOLOv11, 1 clase `Placa`). Para reentrenar con un
dataset en formato YOLO:
```bash
yolo detect train model=yolo11n.pt data=dataset_combinado/data.yaml epochs=100 imgsz=640
```
# IA_radar
