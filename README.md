# Radar de placas — UTA / IA

Sistema de control de velocidad en tiempo real:
**YOLOv11** detecta la placa → **CNN propia** lee los caracteres → **lógica difusa** decide la sanción.

## Stack

| Módulo | Tecnología |
|---|---|
| Detección de placa | YOLOv11 (`best.pt`) |
| OCR de caracteres | CNN ResNet propia (`models/ocr_char.pt`) |
| Velocidad | Background subtraction + 2 líneas virtuales |
| Sanción | scikit-fuzzy — 0–10 km/h felicitación / 10–20 normal / >20 multa en horas |
| Registro | JSON (`registros/eventos.json`) + captura JPG |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Comandos

```bash
# Pipeline completo
.venv/bin/python main.py video videos/carros3.mp4   # archivo de video
.venv/bin/python main.py camara                     # cámara física (índice 0)
.venv/bin/python main.py digital <url>              # DroidCam / IP cam
.venv/bin/python main.py demo                       # sin cámara, datos ficticios

# Leer una placa desde imagen estática
.venv/bin/python cnn/inferencia.py <imagen.jpg>

# Reentrenar el clasificador de caracteres (requiere Dataset_OCR_Placas/)
.venv/bin/python cnn/entrenar_ocr_real.py

# Probar la lógica difusa
.venv/bin/python velocidad/logica_difusa.py
```

## Controles en ventana

| Tecla | Acción |
|---|---|
| `F` | Pantalla completa |
| `R` | Reiniciar ciclo |
| Arrastrar línea A/B | Reposicionar umbrales de velocidad |
| `ESC` | Salir |

## Estructura relevante

```
best.pt                      # Detector YOLOv11 (1 clase: placa)
models/ocr_char.pt           # Clasificador CNN de caracteres (36 clases)
cnn/
  modelo.py                  # Arquitectura CNNPlacas (~1.2 M params)
  entrenar_ocr_real.py       # Entrenamiento con Dataset_OCR_Placas/
  inferencia.py              # YOLO → preproceso → segmentación → CNN → voto
main.py                      # Bucle principal + hilo worker + VotadorPlaca
velocidad/logica_difusa.py   # Sistema difuso de sanción
utils/registro.py            # Log JSON de eventos
docs/informe_radar.pdf       # Informe técnico completo
```

## Informe técnico

`docs/informe_radar.pdf` — arquitectura, métricas, lógica difusa, resultados en videos reales.

```bash
cd docs && latexmk -pdf informe_radar.tex   # recompilar
```
