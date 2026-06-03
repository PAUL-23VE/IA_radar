"""
cnn/inferencia.py
Pipeline de OCR: YOLOv11 detecta la región de placa → segmentación de
caracteres → clasificador CNN (un carácter a la vez, en batch) → texto.

Arquitectura del sistema de OCR:
  1. YOLOv11 (best.pt)        — detección de región de placa (bbox)
  2. Preprocesamiento          — gris + sharpening (si hay blur) + CLAHE
  3. Segmentación              — binarización + componentes conexos → cajas de chars
  4. Clasificador CNN          — models/ocr_char.pt (32×32 → 36 clases), batch
  5. Validación de formato     — regex ecuatoriano ABC-NNNN

El clasificador se entrena con caracteres REALES (Dataset_OCR_Placas) en
`cnn/entrenar_ocr_real.py`. Sin EasyOCR, sin CRNN.
"""

import os
import re
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar")

import sys
sys.path.insert(0, os.path.dirname(__file__))

from modelo import crear_modelo_cnn  # noqa: E402

# ----------------------------------------------------------------
#  Configuración
# ----------------------------------------------------------------
RUTA_YOLO = os.getenv("YOLO_MODEL_PATH", "best.pt")
RUTA_CNN  = os.path.join(os.path.dirname(__file__), "..", "models", "ocr_char.pt")
CONF_PLACA = 0.15

IMG_SIZE = 32  # tamaño de entrada del clasificador (se sobrescribe con el del checkpoint)

# Confianza mínima media de los caracteres para aceptar una lectura.
# El filtro principal de falsos positivos es la validación de formato ABC-NNNN.
CONF_CNN_MIN = 0.35

# Singletons — thread-safe
_lock         = threading.Lock()
_yolo_cache   = None
_cnn_cache    = None
_classes      = None
_device_cache = None


# ----------------------------------------------------------------
#  Carga de modelos (singleton thread-safe)
# ----------------------------------------------------------------

def resolver_dispositivo() -> str:
    global _device_cache
    if _device_cache is None:
        _device_cache = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _device_cache


def cargar_yolo() -> YOLO:
    global _yolo_cache
    if _yolo_cache is None:
        with _lock:
            if _yolo_cache is None:
                _yolo_cache = YOLO(RUTA_YOLO)
    return _yolo_cache


def cargar_cnn():
    """Carga el clasificador de caracteres y su lista de clases."""
    global _cnn_cache, _classes, IMG_SIZE
    if _cnn_cache is None:
        with _lock:
            if _cnn_cache is None:
                dispositivo = resolver_dispositivo()
                modelo = crear_modelo_cnn().to(dispositivo)
                ruta = os.path.abspath(RUTA_CNN)
                if os.path.exists(ruta):
                    ckpt = torch.load(ruta, map_location=dispositivo, weights_only=False)
                    modelo.load_state_dict(ckpt["state_dict"])
                    _classes = ckpt.get("classes",
                                        list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                    IMG_SIZE = ckpt.get("img_size", 32)
                    print(f"[CNN] Modelo cargado: {ruta} ({len(_classes)} clases)")
                else:
                    print(f"[CNN] ADVERTENCIA: modelo no encontrado en {ruta}")
                    print("       Ejecuta: .venv/bin/python cnn/entrenar_ocr_real.py")
                    _classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                modelo.eval()
                _cnn_cache = modelo
    return _cnn_cache


# ----------------------------------------------------------------
#  1. Detección de placa (YOLOv11)
# ----------------------------------------------------------------

def detectar_region_placa(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    yolo        = cargar_yolo()
    dispositivo = resolver_dispositivo()

    usar_half = "cuda" in dispositivo
    res = yolo(frame, conf=CONF_PLACA, verbose=False, half=usar_half)[0]

    if not res.boxes:
        return None, None

    mejor_box  = None
    mejor_conf = -1.0
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf > mejor_conf:
            mejor_conf = conf
            mejor_box  = box.xywh[0]

    if mejor_box is None:
        return None, None

    x_c, y_c, w, h = map(int, mejor_box)
    x = x_c - w // 2
    y = y_c - h // 2

    # Padding 8% para no cortar bordes de letras exteriores
    px = max(2, int(w * 0.08))
    py = max(2, int(h * 0.08))

    y1 = max(0, y - py);  y2 = min(frame.shape[0], y + h + py)
    x1 = max(0, x - px);  x2 = min(frame.shape[1], x + w + px)

    recorte = frame[y1:y2, x1:x2]
    return recorte, (x1, y1, x2 - x1, y2 - y1)


# ----------------------------------------------------------------
#  2. Preprocesamiento del recorte
# ----------------------------------------------------------------

def _sharpening_kernel() -> np.ndarray:
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def _preparar_gris(recorte: np.ndarray) -> np.ndarray:
    """Convierte crop a gris, aplica sharpening si hay blur y normaliza con CLAHE."""
    if recorte.ndim == 3:
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    else:
        gris = recorte.copy()

    # Sharpening si hay motion blur (Laplacian variance baja)
    if cv2.Laplacian(gris, cv2.CV_64F).var() < 80:
        gris = cv2.filter2D(gris, -1, _sharpening_kernel())

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gris)


# ----------------------------------------------------------------
#  3. Segmentación de caracteres
# ----------------------------------------------------------------

def _segmentar_caracteres(gris: np.ndarray) -> list[np.ndarray]:
    """
    Segmenta los caracteres de un recorte de placa en escala de grises.

    Estrategia:
      - reescala a altura fija (mejor estabilidad de los umbrales)
      - binariza con Otsu, asegurando que el texto sea blanco (minoría)
      - componentes conexos filtrados por alto/aspecto/área
      - ordena por X y devuelve los recortes EN GRIS (no binarios), que es
        el dominio con el que se entrenó el clasificador.

    Returns:
        lista de recortes en gris (uint8), ordenados de izquierda a derecha.
    """
    H_OBJ = 64
    h0, w0 = gris.shape[:2]
    if h0 == 0 or w0 == 0:
        return []
    escala = H_OBJ / h0
    g = cv2.resize(gris, (max(1, int(w0 * escala)), H_OBJ), interpolation=cv2.INTER_LINEAR)
    H, W = g.shape

    # Umbral ADAPTATIVO: separa caracteres bajo iluminación irregular y blur,
    # donde Otsu global fusionaba toda la placa en un solo blob. Texto oscuro
    # sobre fondo claro → foreground (texto) BLANCO.
    binimg = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9,
    )
    if np.count_nonzero(binimg) > binimg.size * 0.5:   # placa con texto claro/fondo oscuro
        binimg = cv2.bitwise_not(binimg)

    # Despeckle (quita ruido sal/pimienta sin unir caracteres adyacentes)
    binimg = cv2.morphologyEx(
        binimg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)

    cajas = []
    for i in range(1, n):  # 0 = fondo
        x, y, w, h, area = stats[i]
        # Altura típica de carácter (el recorte de YOLO suele traer margen, por
        # eso el umbral inferior es bajo; los outliers se filtran luego por mediana)
        if h < 0.25 * H or h > 0.95 * H:
            continue
        rel = w / h
        if rel < 0.10 or rel > 1.3:           # descartar barras/ruido y bloques anchos
            continue
        if area < 0.008 * H * W:
            continue
        cajas.append((x, y, w, h))

    if not cajas:
        return []

    # Filtrar outliers de altura respecto a la mediana (ruido, marcos)
    alturas = np.array([c[3] for c in cajas], dtype=np.float32)
    h_med   = float(np.median(alturas))
    cajas   = [c for c in cajas if 0.6 * h_med <= c[3] <= 1.4 * h_med]
    cajas.sort(key=lambda c: c[0])  # orden izquierda → derecha

    recortes = []
    for x, y, w, h in cajas:
        m = max(1, int(0.12 * h))             # padding alrededor del carácter
        y1 = max(0, y - m); y2 = min(H, y + h + m)
        x1 = max(0, x - m); x2 = min(W, x + w + m)
        recortes.append(g[y1:y2, x1:x2])
    return recortes


# ----------------------------------------------------------------
#  4. Clasificación CNN (batch)
# ----------------------------------------------------------------

def _clasificar_caracteres(recortes: list[np.ndarray]) -> np.ndarray:
    """
    Clasifica una lista de recortes de carácter en un solo batch GPU.

    Returns:
        probs: matriz numpy [N, 36] de probabilidades softmax por carácter.
    """
    if not recortes:
        return np.zeros((0, len(_classes or [0] * 36)), dtype=np.float32)

    modelo      = cargar_cnn()
    dispositivo = resolver_dispositivo()

    batch = np.zeros((len(recortes), 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, r in enumerate(recortes):
        ch = cv2.resize(r, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        batch[i, 0] = ch.astype(np.float32) / 255.0

    tensor = torch.from_numpy(batch).to(dispositivo)
    with torch.no_grad():
        logits = modelo(tensor)
        probs  = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


# Índices de clases de letras (A-Z) y dígitos (0-9), según el orden del checkpoint
def _indices_grupos() -> tuple[list[int], list[int]]:
    letras  = [i for i, c in enumerate(_classes) if c.isalpha()]
    digitos = [i for i, c in enumerate(_classes) if c.isdigit()]
    return letras, digitos


def _decodificar_posicional(probs: np.ndarray) -> tuple[str, float]:
    """
    Decodifica explotando el formato ecuatoriano ABC-NNNN: las primeras 3
    posiciones son letras y las restantes dígitos. Restringir cada carácter a
    su grupo elimina las confusiones cruzadas O↔0, I↔1, Z↔2, B↔8, S↔5
    (las más frecuentes), aumentando mucho la precisión a nivel de placa.

    Solo aplica cuando hay 6 o 7 caracteres segmentados (placa EC válida).
    Retorna ("", 0.0) si no aplica.
    """
    n = probs.shape[0]
    if n not in (6, 7):
        return "", 0.0

    letras, digitos = _indices_grupos()
    chars, confs = [], []
    for i in range(n):
        grupo = letras if i < 3 else digitos
        j     = grupo[int(np.argmax(probs[i, grupo]))]
        chars.append(_classes[j])
        confs.append(float(probs[i, j]))

    return "".join(chars), float(np.mean(confs))


def _decodificar_plano(probs: np.ndarray) -> tuple[str, float]:
    """Decodificación simple: argmax por carácter (fallback)."""
    if probs.shape[0] == 0:
        return "", 0.0
    idxs  = probs.argmax(axis=1)
    confs = probs[np.arange(probs.shape[0]), idxs]
    return "".join(_classes[i] for i in idxs), float(confs.mean())


# ----------------------------------------------------------------
#  5. Validación de formato (placa ecuatoriana)
# ----------------------------------------------------------------

def _validar_formato(texto: str) -> str:
    """
    Valida y formatea: 'GTN5618' → 'GTN-5618'.
    Acepta 6 o 7 chars (3 letras + 3-4 dígitos). Retorna '' si no cumple.
    """
    texto = texto.upper().replace("-", "").replace(" ", "")
    for largo in (7, 6):
        for inicio in range(len(texto) - largo + 1):
            cand    = texto[inicio: inicio + largo]
            placa   = f"{cand[:3]}-{cand[3:]}"
            if re.match(r"^[A-Z]{3}-\d{3,4}$", placa):
                return placa
    return ""


# ----------------------------------------------------------------
#  Lectura completa de una placa
# ----------------------------------------------------------------

def leer_placa_cnn(recorte: np.ndarray) -> tuple[str, str, float]:
    """
    Lee el texto de una placa desde su recorte (salida de YOLO).

    Returns:
        (placa_formateada, texto_crudo, confianza)
    """
    gris  = _preparar_gris(recorte)
    chars = _segmentar_caracteres(gris)
    probs = _clasificar_caracteres(chars)

    # 1º intento: decodificación posicional (formato EC ABC-NNNN)
    texto_pos, conf_pos = _decodificar_posicional(probs)
    placa = _validar_formato(texto_pos)
    if placa:
        if conf_pos < CONF_CNN_MIN:
            placa = ""
        return placa, texto_pos, conf_pos

    # Fallback: argmax plano + búsqueda de formato por ventana deslizante
    texto, conf = _decodificar_plano(probs)
    placa = _validar_formato(texto)
    if placa and conf < CONF_CNN_MIN:
        placa = ""
    return placa, texto, conf


# ----------------------------------------------------------------
#  Interfaz pública
# ----------------------------------------------------------------

def leer_placa_desde_recorte(recorte: np.ndarray, max_variantes=None) -> tuple[str, str, float]:
    """Wrapper de compatibilidad con scripts existentes."""
    return leer_placa_cnn(recorte)


def reconocer_placa(frame: np.ndarray, max_variantes: int | None = None
                    ) -> tuple[str, tuple | None, float]:
    """
    Pipeline principal: frame → YOLO → recorte → segmentación → CNN
    → (placa, bbox, conf).
    """
    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        return "", None, 0.0
    placa, _texto, conf = leer_placa_cnn(recorte)
    return placa, bbox, conf


# ----------------------------------------------------------------
#  Prueba con imagen estática
# ----------------------------------------------------------------
if __name__ == "__main__":
    ruta  = sys.argv[1] if len(sys.argv) > 1 else "prueba_placa.jpg"
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"No se pudo cargar: {ruta}")
        sys.exit(1)

    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        print("No se detectó placa.")
        sys.exit(0)

    placa, texto, conf = leer_placa_cnn(recorte)
    print(f"Texto crudo: {texto!r}")
    print(f"Placa: {placa!r}  Confianza: {conf:.3f}")

    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{placa or texto or '?'} ({conf:.2f})",
                    (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("resultado.jpg", frame)
        print("Guardado en resultado.jpg")
