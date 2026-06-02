"""
cnn/inferencia.py
Pipeline completo de reconocimiento de placas: YOLOv11 + CNN Propia (PyTorch).

Pipeline:
  1. YOLOv11 (best.pt) detecta la región de la placa
  2. Preprocesamiento: sharpening + CLAHE para imágenes de cámara móvil
  3. Segmentador multi-estrategia con bounds superior E INFERIOR
     (evita incluir texto "ECUADOR" y pie de placa)
  4. Predicción posicional: pos 0-2 → letras, pos 3-6 → dígitos
  5. Fallback EasyOCR si CNN falla o confianza < umbral
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
from modelo import CLASES, NUM_CLASES, crear_modelo_cnn

# ----------------------------------------------------------------
#  Índices de clases
# ----------------------------------------------------------------
IDX_LETRAS  = list(range(26))       # A..Z
IDX_DIGITOS = list(range(26, 36))   # 0..9
OBJETIVO    = {6, 7}                # longitud de placa sin guión

# ----------------------------------------------------------------
#  Configuración
# ----------------------------------------------------------------
RUTA_YOLO  = os.getenv("YOLO_MODEL_PATH", "best.pt")
CONF_PLACA = 0.15

# Singletons — carga diferida y thread-safe
_lock         = threading.Lock()
_yolo_cache   = None
_cnn_cache    = None
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
    global _cnn_cache
    if _cnn_cache is None:
        with _lock:
            if _cnn_cache is None:
                dispositivo = resolver_dispositivo()
                m = crear_modelo_cnn().to(dispositivo)
                ruta_pt = os.path.join(
                    os.path.dirname(__file__), "..", "models", "modelo_entrenado.pt"
                )
                m.load_state_dict(
                    torch.load(ruta_pt, map_location=dispositivo, weights_only=True)
                )
                m.eval()
                _cnn_cache = m
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

    y1 = max(0, y - py)
    y2 = min(frame.shape[0], y + h + py)
    x1 = max(0, x - px)
    x2 = min(frame.shape[1], x + w + px)

    recorte = frame[y1:y2, x1:x2]
    return recorte, (x1, y1, x2 - x1, y2 - y1)


# ================================================================
#  2. Preprocesamiento de la región de placa
# ================================================================

def _sharpening_kernel() -> np.ndarray:
    """Kernel de enfoque laplaciano suave."""
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def _preparar_recorte(recorte: np.ndarray) -> np.ndarray:
    """
    Escala el recorte y aplica sharpening + denoise.
    Optimizado para imágenes de cámara móvil con motion blur.
    """
    h_r, w_r = recorte.shape[:2]

    # Escalar a mínimo 80px alto × 200px ancho para mejorar segmentación
    ALTO_MIN, ANCHO_MIN = 80, 200
    if h_r < ALTO_MIN or w_r < ANCHO_MIN:
        escala  = max(ALTO_MIN / h_r, ANCHO_MIN / w_r)
        nuevo_w = int(w_r * escala)
        nuevo_h = int(h_r * escala)
        recorte = cv2.resize(recorte, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)

    # Medir borrosidad (Laplacian variance); si < 80 → afilar
    gris_test = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gris_test, cv2.CV_64F).var()
    if blur_score < 80:
        recorte = cv2.filter2D(recorte, -1, _sharpening_kernel())

    return recorte


# ================================================================
#  3. Segmentación de caracteres
# ================================================================

def _binarizar(gris: np.ndarray, modo: str) -> np.ndarray:
    if modo == "adaptativa":
        return cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10,
        )
    elif modo == "adaptativa_suave":
        return cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 7,
        )
    elif modo == "otsu":
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return b
    elif modo == "otsu_normal":
        _, b = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    elif modo == "umbral_fijo":
        _, b = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
        return b
    return gris


def _extraer_cajas(binaria: np.ndarray, h_zona: int) -> list[tuple]:
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilatada = cv2.dilate(binaria, kernel, iterations=1)
    contornos, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        ar   = float(w) / h if h > 0 else 0
        area = w * h
        # Umbral altura más estricto (0.35 vs 0.28) para filtrar "ECUADOR" pequeño
        if (0.06 < ar < 1.6
                and h > h_zona * 0.35
                and h < h_zona * 0.99
                and area > 20):
            cajas.append((x, y, w, h))

    cajas.sort(key=lambda b: b[0])

    # NMS: suprimir duplicados solapados
    filtradas = []
    for caja in cajas:
        x1, y1, w1, h1 = caja
        cx1 = x1 + w1 // 2
        solapada = any(
            abs(cx1 - (x2 + w2 // 2)) < max(w1, w2) * 0.4
            for x2, y2, w2, h2 in filtradas
        )
        if not solapada:
            filtradas.append(caja)
    return filtradas


def _split_anchos(cajas: list, binaria: np.ndarray, h_zona: int) -> list[tuple]:
    if len(cajas) < 2:
        return cajas
    anchos    = [w for _, _, w, _ in cajas]
    ancho_med = np.median(anchos)
    resultado = []
    for (x, y, w, h) in cajas:
        if w > ancho_med * 1.7 and w > 10:
            mitad = w // 2
            resultado.append((x,        y, mitad,     h))
            resultado.append((x + mitad, y, w - mitad, h))
        else:
            resultado.append((x, y, w, h))
    resultado.sort(key=lambda b: b[0])
    return resultado


def _proyeccion_vertical(zona_bin: np.ndarray) -> list[tuple]:
    h, w = zona_bin.shape
    proj  = zona_bin.sum(axis=0).astype(float)
    if proj.max() == 0:
        return []
    proj /= proj.max()
    k     = max(1, w // 40)
    proj  = np.convolve(proj, np.ones(k) / k, mode="same")

    activo = (proj > 0.06).astype(np.uint8)
    segmentos = []
    en_seg = False
    ini = 0
    for col in range(w):
        if activo[col] and not en_seg:
            ini = col; en_seg = True
        elif not activo[col] and en_seg:
            fin = col
            seg = zona_bin[:, ini:fin]
            rows = np.where(seg.sum(axis=1) > 0)[0]
            if len(rows) > 0:
                y_s = int(rows.min()); h_s = int(rows.max()) - y_s + 1
                w_s = fin - ini
                ar  = w_s / h_s if h_s > 0 else 0
                if 0.05 < ar < 1.8 and h_s > h * 0.30:
                    segmentos.append((ini, y_s, w_s, h_s))
            en_seg = False
    if en_seg:
        fin = w
        seg = zona_bin[:, ini:fin]
        rows = np.where(seg.sum(axis=1) > 0)[0]
        if len(rows) > 0:
            y_s = int(rows.min()); h_s = int(rows.max()) - y_s + 1
            w_s = fin - ini
            ar  = w_s / h_s if h_s > 0 else 0
            if 0.05 < ar < 1.8 and h_s > h * 0.30:
                segmentos.append((ini, y_s, w_s, h_s))
    return segmentos


def _distancia_objetivo(n: int) -> int:
    return 0 if n in OBJETIVO else min(abs(n - t) for t in OBJETIVO)


def _cajas_a_imagenes(cajas: list, binaria: np.ndarray) -> list[np.ndarray]:
    imagenes = []
    for (x, y, w, h) in cajas:
        char_img = binaria[y: y + h, x: x + w]
        lado     = max(h, w) + 8
        padded   = np.zeros((lado, lado), dtype=np.uint8)
        y_off = (lado - h) // 2
        x_off = (lado - w) // 2
        padded[y_off: y_off + h, x_off: x_off + w] = char_img
        imagenes.append(cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA))
    return imagenes


def segmentar_caracteres(recorte: np.ndarray) -> list[np.ndarray]:
    """
    Motor de segmentación multi-estrategia:
      - Preprocesa la imagen (sharpening para DroidCam)
      - Grid de 5 zonas (top,bot) × 5 modos × 2 inversiones
      - Zona con bounds superior e inferior para excluir "ECUADOR" y texto pie
      - Split automático de boxes anchos
      - Proyección vertical como fallback
    """
    if recorte is None or recorte.size == 0:
        return []

    recorte = _preparar_recorte(recorte)
    h_r, w_r = recorte.shape[:2]

    mejor_cajas = []
    mejor_bin   = None
    mejor_dist  = float("inf")

    # (fracción_top, fracción_bot): zona donde están los caracteres principales.
    # Los bounds inferiores evitan incluir el texto pequeño al pie de la placa.
    ZONAS = [
        (0.20, 0.95),
        (0.15, 0.95),
        (0.25, 0.92),
        (0.30, 0.95),
        (0.10, 0.90),
        (0.35, 0.98),
    ]
    MODOS = ["adaptativa", "adaptativa_suave", "otsu", "otsu_normal", "umbral_fijo"]

    for top_pct, bot_pct in ZONAS:
        y_top = int(h_r * top_pct)
        y_bot = int(h_r * bot_pct)
        zona  = recorte[y_top:y_bot, :]
        h_z   = zona.shape[0]
        if h_z < 10:
            continue

        gris  = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
        gris  = clahe.apply(gris)
        gris  = cv2.GaussianBlur(gris, (3, 3), 0)

        for modo in MODOS:
            for invertir in (False, True):
                binaria = _binarizar(gris, modo)
                if invertir:
                    binaria = cv2.bitwise_not(binaria)

                cajas = _extraer_cajas(binaria, h_z)

                if len(cajas) < 6:
                    cajas_split = _split_anchos(cajas, binaria, h_z)
                    if _distancia_objetivo(len(cajas_split)) < _distancia_objetivo(len(cajas)):
                        cajas = cajas_split

                if len(cajas) > 7:
                    cajas_top = sorted(cajas, key=lambda b: b[2] * b[3], reverse=True)[:7]
                    cajas_top.sort(key=lambda b: b[0])
                    if _distancia_objetivo(len(cajas_top)) <= _distancia_objetivo(len(cajas)):
                        cajas = cajas_top

                dist = _distancia_objetivo(len(cajas))
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_cajas = cajas
                    mejor_bin   = binaria

                if mejor_dist == 0:
                    break
            if mejor_dist == 0:
                break
        if mejor_dist == 0:
            break

    # ── Fallback: proyección vertical ──────────────────────────
    if mejor_dist > 0:
        for top_pct, bot_pct in ZONAS:
            y_top = int(h_r * top_pct)
            y_bot = int(h_r * bot_pct)
            zona  = recorte[y_top:y_bot, :]
            h_z   = zona.shape[0]
            if h_z < 10:
                continue
            gris  = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
            gris  = clahe.apply(gris)
            gris  = cv2.GaussianBlur(gris, (3, 3), 0)

            for modo in ["adaptativa", "otsu"]:
                for invertir in (False, True):
                    binaria     = _binarizar(gris, modo)
                    if invertir:
                        binaria = cv2.bitwise_not(binaria)

                    cajas_proy = _proyeccion_vertical(binaria)
                    if not cajas_proy:
                        continue

                    dist = _distancia_objetivo(len(cajas_proy))
                    if dist < mejor_dist:
                        mejor_dist  = dist
                        mejor_cajas = cajas_proy
                        mejor_bin   = binaria

                    if mejor_dist == 0:
                        break
                if mejor_dist == 0:
                    break
            if mejor_dist == 0:
                break

    if mejor_bin is None or not mejor_cajas:
        return []

    return _cajas_a_imagenes(mejor_cajas, mejor_bin)


# ----------------------------------------------------------------
#  4. Predicción posicional con TTA ligero
#     Pos 0-2 → letras | Pos 3-6 → dígitos
# ----------------------------------------------------------------
def _variante_char(img: np.ndarray) -> np.ndarray:
    """Versión CLAHE del carácter para TTA."""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    return clahe.apply(img)


def predecir_caracteres(imagenes: list[np.ndarray]) -> tuple[str, float]:
    """
    Inferencia posicional con TTA ligero (original + CLAHE).
    Las primeras 3 posiciones se restringen a letras, las siguientes a dígitos.
    """
    if not imagenes:
        return "", 0.0

    modelo      = cargar_cnn()
    dispositivo = resolver_dispositivo()

    def _build_batch(imgs):
        return torch.cat([
            torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            for im in imgs
        ], dim=0).to(dispositivo)

    with torch.no_grad():
        # TTA: original + variante CLAHE → promedio de logits
        batch_orig  = _build_batch(imagenes)
        batch_clahe = _build_batch([_variante_char(im) for im in imagenes])
        logits = (modelo(batch_orig) + modelo(batch_clahe)) * 0.5

        texto     = ""
        suma_conf = 0.0
        for i, row in enumerate(logits):
            mascara = torch.full((NUM_CLASES,), float("-inf"), device=dispositivo)
            if i < 3:
                mascara[IDX_LETRAS]  = row[IDX_LETRAS]
            else:
                mascara[IDX_DIGITOS] = row[IDX_DIGITOS]

            probs = F.softmax(mascara, dim=0)
            idx   = int(probs.argmax())
            texto += CLASES[idx]
            suma_conf += probs[idx].item()

    return texto, suma_conf / len(imagenes)


# ----------------------------------------------------------------
#  5. Validación tolerante (formato ecuatoriano: ABC-1234)
# ----------------------------------------------------------------
def validar_y_corregir_placa(placa_cruda: str) -> str:
    if len(placa_cruda) < 6 or len(placa_cruda) > 9:
        return ""

    for largo in (7, 6):
        for inicio in range(len(placa_cruda) - largo + 1):
            candidato = placa_cruda[inicio: inicio + largo]
            letras    = candidato[:3]
            numeros   = candidato[3:]
            placa     = f"{letras}-{numeros}"
            if re.match(r"^[A-Z]{3}-\d{3,4}$", placa):
                return placa
    return ""


# ----------------------------------------------------------------
#  Fallback: EasyOCR (LSTM)
# ----------------------------------------------------------------
_easyocr_reader = None


def get_easyocr_placa(recorte: np.ndarray) -> str:
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["es"], gpu=True, verbose=False)

    h, w = recorte.shape[:2]
    if h < 80 or w < 200:
        escala  = max(80 / h, 200 / w)
        recorte = cv2.resize(
            recorte,
            (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Afilar para LSTM
    recorte = cv2.filter2D(recorte, -1, _sharpening_kernel())

    import unicodedata
    res   = _easyocr_reader.readtext(
        recorte,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
        paragraph=False,
    )
    texto = " ".join(t for _, t, _ in res)
    texto = unicodedata.normalize("NFD", texto.upper())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    m     = re.search(r"([A-Z]{3})[\s\-.]?(\d{3,4})", texto)
    return f"{m.group(1)}-{m.group(2)}" if m else ""


# ----------------------------------------------------------------
#  Interfaz pública
# ----------------------------------------------------------------
def leer_placa_desde_recorte(
    recorte: np.ndarray, max_variantes=None
) -> tuple[str, str, float]:
    chars       = segmentar_caracteres(recorte)
    texto, conf = predecir_caracteres(chars)
    placa       = validar_y_corregir_placa(texto)

    # Fallback EasyOCR cuando CNN falla O confianza < 0.80.
    # EasyOCR (LSTM) es superior para placas reales con motion blur o baja resolución.
    # Con threading, el LSTM corre en background sin bloquear la captura de frames.
    if not placa or conf < 0.80:
        placa_ocr = get_easyocr_placa(recorte)
        if placa_ocr:
            placa = placa_ocr
            texto = placa_ocr.replace("-", "")
            conf  = 0.75

    return placa, texto, conf


def reconocer_placa(
    frame: np.ndarray, max_variantes: int | None = None
) -> tuple[str, tuple | None, float]:
    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        return "", None, 0.0
    placa, _texto, _conf = leer_placa_desde_recorte(recorte)
    return placa, bbox, _conf


# ----------------------------------------------------------------
#  Prueba con imagen estática
# ----------------------------------------------------------------
if __name__ == "__main__":
    ruta  = sys.argv[1] if len(sys.argv) > 1 else "prueba_placa.jpg"
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"No se pudo cargar: {ruta}")
    else:
        placa, bbox, conf = reconocer_placa(frame)
        print(f"Placa: {placa!r}  Confianza: {conf:.2f}")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, placa or "?", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("resultado.jpg", frame)
            print("Guardado en resultado.jpg")
