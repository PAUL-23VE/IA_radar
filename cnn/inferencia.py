"""
cnn/inferencia.py
Reconocimiento de placas basado en YOLOv11 + CNN Propia (PyTorch).

Pipeline:
  1. YOLOv11 (best.pt) detecta la región de la placa en el frame
  2. Segmentador multi-estrategia:
       - Grid de zona×binarización×inversión (30 combinaciones)
       - Split automático de boxes muy anchos (letras fusionadas)
       - Proyección vertical como fallback matemático
  3. Predicción posicional: posiciones 0-2 → solo letras (A-Z),
     posiciones 3-6 → solo dígitos (0-9), usando logits directos
  4. Validación tolerante: acepta 5-8 chars y busca la mejor ventana
"""

import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from modelo import crear_modelo_cnn, CLASES, NUM_CLASES

# ----------------------------------------------------------------
#  ÍNDICES DE CLASES
#  CLASES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#  Letras: índices 0-25 | Dígitos: índices 26-35
# ----------------------------------------------------------------
IDX_LETRAS  = list(range(26))       # A..Z
IDX_DIGITOS = list(range(26, 36))   # 0..9
OBJETIVO    = {6, 7}                # longitud de placa sin guión

# ----------------------------------------------------------------
#  CONFIGURACIÓN
# ----------------------------------------------------------------
RUTA_YOLO  = os.getenv("YOLO_MODEL_PATH", "best.pt")
CONF_PLACA = 0.35

_yolo_cache   = None
_cnn_cache    = None
_device_cache = None


# ----------------------------------------------------------------
#  CARGA DE MODELOS (singleton)
# ----------------------------------------------------------------
def resolver_dispositivo() -> str:
    global _device_cache
    if _device_cache is None:
        _device_cache = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _device_cache


def cargar_yolo():
    global _yolo_cache
    if _yolo_cache is None:
        _yolo_cache = YOLO(RUTA_YOLO)
    return _yolo_cache


def cargar_cnn():
    global _cnn_cache
    if _cnn_cache is None:
        dispositivo = resolver_dispositivo()
        _cnn_cache = crear_modelo_cnn().to(dispositivo)
        ruta_pt = os.path.join(
            os.path.dirname(__file__), "..", "models", "modelo_entrenado.pt"
        )
        _cnn_cache.load_state_dict(
            torch.load(ruta_pt, map_location=dispositivo, weights_only=True)
        )
        _cnn_cache.eval()
    return _cnn_cache


# ----------------------------------------------------------------
#  1. DETECCIÓN DE PLACA (YOLOv11)
# ----------------------------------------------------------------
def detectar_region_placa(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    if frame is None or frame.size == 0:
        return None, None

    modelo    = cargar_yolo()
    device    = resolver_dispositivo()
    resultados = modelo(frame, conf=CONF_PLACA, verbose=False, device=device)
    cajas     = resultados[0].boxes

    if cajas is None or len(cajas) == 0:
        return None, None

    mejor = int(cajas.conf.argmax())
    x1, y1, x2, y2 = cajas.xyxy[mejor].cpu().numpy().astype(int)

    h_f, w_f = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f, x2), min(h_f, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None

    recorte = frame[y1:y2, x1:x2]
    bbox    = (x1, y1, x2 - x1, y2 - y1)
    return recorte, bbox


# ================================================================
#  2. SEGMENTACIÓN DE CARACTERES — MOTOR MEJORADO
# ================================================================

def _binarizar(gris: np.ndarray, modo: str) -> np.ndarray:
    """Devuelve imagen binarizada según el modo (letras blancas / fondo negro)."""
    if modo == "adaptativa":
        return cv2.adaptiveThreshold(
            gris, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )
    elif modo == "adaptativa_suave":
        return cv2.adaptiveThreshold(
            gris, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 7
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
    """Extrae bounding-boxes candidatos a caracteres."""
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilatada = cv2.dilate(binaria, kernel, iterations=1)
    contornos, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        ar   = float(w) / h if h > 0 else 0
        area = w * h
        if (0.06 < ar < 1.6
                and h > h_zona * 0.28
                and h < h_zona * 0.99
                and area > 15):
            cajas.append((x, y, w, h))

    # Ordenar izquierda → derecha
    cajas.sort(key=lambda b: b[0])

    # Suprimir duplicados solapados (NMS)
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
    """
    Divide bounding-boxes inusualmente anchos (letras fusionadas).
    Heurística: si un box tiene el doble del ancho promedio, lo parte por la mitad.
    """
    if len(cajas) < 2:
        return cajas

    anchos = [w for _, _, w, _ in cajas]
    ancho_med = np.median(anchos)

    resultado = []
    for (x, y, w, h) in cajas:
        if w > ancho_med * 1.7 and w > 10:
            # Dividir en dos mitades iguales
            mitad = w // 2
            resultado.append((x,        y, mitad,     h))
            resultado.append((x + mitad, y, w - mitad, h))
        else:
            resultado.append((x, y, w, h))
    resultado.sort(key=lambda b: b[0])
    return resultado


def _proyeccion_vertical(zona_bin: np.ndarray) -> list[tuple]:
    """
    Fallback: suma de píxeles por columna → encuentra los 'valles'
    que separan caracteres. Muy robusto para texto horizontal limpio.
    """
    h, w = zona_bin.shape
    proj = zona_bin.sum(axis=0).astype(float)

    # Normalizar y suavizar
    if proj.max() == 0:
        return []
    proj = proj / proj.max()
    kernel_s = np.ones(max(1, w // 40)) / max(1, w // 40)
    proj_s = np.convolve(proj, kernel_s, mode='same')

    umbral = 0.06
    activo = (proj_s > umbral).astype(np.uint8)

    # Encontrar inicio y fin de cada segmento activo
    segmentos = []
    en_seg = False
    ini = 0
    for col in range(w):
        if activo[col] and not en_seg:
            ini = col
            en_seg = True
        elif not activo[col] and en_seg:
            fin = col
            seg_bin = zona_bin[:, ini:fin]
            rows = np.where(seg_bin.sum(axis=1) > 0)[0]
            if len(rows) > 0:
                y_s = int(rows.min())
                h_s = int(rows.max()) - y_s + 1
                w_s = fin - ini
                ar  = w_s / h_s if h_s > 0 else 0
                if 0.05 < ar < 1.8 and h_s > h * 0.25:
                    segmentos.append((ini, y_s, w_s, h_s))
            en_seg = False
    if en_seg:
        fin = w
        seg_bin = zona_bin[:, ini:fin]
        rows = np.where(seg_bin.sum(axis=1) > 0)[0]
        if len(rows) > 0:
            y_s = int(rows.min())
            h_s = int(rows.max()) - y_s + 1
            w_s = fin - ini
            ar  = w_s / h_s if h_s > 0 else 0
            if 0.05 < ar < 1.8 and h_s > h * 0.25:
                segmentos.append((ini, y_s, w_s, h_s))

    return segmentos


def _distancia_objetivo(n: int) -> int:
    return 0 if n in OBJETIVO else min(abs(n - t) for t in OBJETIVO)


def _cajas_a_imagenes(cajas: list, binaria: np.ndarray) -> list[np.ndarray]:
    """Recorta, cuadra con padding y escala cada carácter a 32×32."""
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
      - Grid de 5 zonas × 5 modos × 2 inversiones = 50 combinaciones
      - Split automático de boxes anchos (letras pegadas)
      - Proyección vertical como fallback
      - Devuelve la combinación más cercana a 6-7 caracteres
    """
    if recorte is None or recorte.size == 0:
        return []

    h_r, w_r = recorte.shape[:2]

    mejor_cajas   = []
    mejor_bin     = None
    mejor_zona_pct = 0.18
    mejor_dist    = float("inf")

    ZONAS  = [0.18, 0.10, 0.25, 0.00, 0.30]
    MODOS  = ["adaptativa", "adaptativa_suave", "otsu", "otsu_normal", "umbral_fijo"]

    for zona_pct in ZONAS:
        zona = recorte[int(h_r * zona_pct):, :]
        h_z  = zona.shape[0]
        if h_z < 8:
            continue

        gris  = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gris  = clahe.apply(gris)
        gris  = cv2.GaussianBlur(gris, (3, 3), 0)

        for modo in MODOS:
            for invertir in (False, True):
                binaria = _binarizar(gris, modo)
                if invertir:
                    binaria = cv2.bitwise_not(binaria)

                cajas = _extraer_cajas(binaria, h_z)

                # Intentar split si faltan chars
                if len(cajas) < 6:
                    cajas_split = _split_anchos(cajas, binaria, h_z)
                    if _distancia_objetivo(len(cajas_split)) < _distancia_objetivo(len(cajas)):
                        cajas = cajas_split

                # Recortar exceso si sobran
                if len(cajas) > 7:
                    cajas_top = sorted(cajas, key=lambda b: b[2] * b[3], reverse=True)[:7]
                    cajas_top.sort(key=lambda b: b[0])
                    if _distancia_objetivo(len(cajas_top)) <= _distancia_objetivo(len(cajas)):
                        cajas = cajas_top

                dist = _distancia_objetivo(len(cajas))
                if dist < mejor_dist:
                    mejor_dist    = dist
                    mejor_cajas   = cajas
                    mejor_bin     = binaria
                    mejor_zona_pct = zona_pct

                if mejor_dist == 0:
                    break
            if mejor_dist == 0:
                break
        if mejor_dist == 0:
            break

    # ── Fallback: proyección vertical ───────────────────────────
    if mejor_dist > 0:
        for zona_pct in ZONAS:
            zona = recorte[int(h_r * zona_pct):, :]
            h_z  = zona.shape[0]
            if h_z < 8:
                continue
            gris  = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gris  = clahe.apply(gris)
            gris  = cv2.GaussianBlur(gris, (3, 3), 0)

            for modo in ["adaptativa", "otsu"]:
                for invertir in (False, True):
                    binaria = _binarizar(gris, modo)
                    if invertir:
                        binaria = cv2.bitwise_not(binaria)

                    cajas_proy = _proyeccion_vertical(binaria)
                    if not cajas_proy:
                        continue

                    dist = _distancia_objetivo(len(cajas_proy))
                    if dist < mejor_dist:
                        mejor_dist    = dist
                        mejor_cajas   = cajas_proy
                        mejor_bin     = binaria
                        mejor_zona_pct = zona_pct

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
#  3. PREDICCIÓN POSICIONAL
#     Posiciones 0-2 → clase letra (idx 0-25)
#     Posiciones 3-6 → clase dígito (idx 26-35)
# ----------------------------------------------------------------
def predecir_caracteres(imagenes: list[np.ndarray]) -> tuple[str, float]:
    """
    Inferencia posicional: las primeras 3 posiciones se restringen
    a letras y las siguientes a dígitos, usando los logits directamente.
    """
    if not imagenes:
        return "", 0.0

    modelo      = cargar_cnn()
    dispositivo = resolver_dispositivo()

    tensores = [
        torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        for img in imagenes
    ]
    batch = torch.cat(tensores, dim=0).to(dispositivo)

    texto     = ""
    suma_conf = 0.0

    with torch.no_grad():
        logits = modelo(batch)

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
#  4. VALIDACIÓN TOLERANTE
#     Acepta strings de 5-8 chars y busca la mejor ventana de 6-7
# ----------------------------------------------------------------
def validar_y_corregir_placa(placa_cruda: str) -> str:
    """
    Intenta armar una placa válida (ABC-NNNN) a partir de la cadena cruda.
    Prueba todas las ventanas de 6 y 7 caracteres.
    """
    if len(placa_cruda) < 5 or len(placa_cruda) > 9:
        return ""

    for largo in (7, 6):
        # Deslizar una ventana de 'largo' caracteres por la cadena
        for inicio in range(len(placa_cruda) - largo + 1):
            candidato = placa_cruda[inicio: inicio + largo]
            letras  = candidato[:3]
            numeros = candidato[3:]
            placa   = f"{letras}-{numeros}"
            if re.match(r"^[A-Z]{3}-\d{3,4}$", placa):
                return placa
    return ""


# ----------------------------------------------------------------
#  INTERFAZ PÚBLICA (compatibilidad con scripts existentes)
# ----------------------------------------------------------------
def leer_placa_desde_recorte(
    recorte: np.ndarray, max_variantes=None
) -> tuple[str, str, float]:
    chars       = segmentar_caracteres(recorte)
    texto, conf = predecir_caracteres(chars)
    placa       = validar_y_corregir_placa(texto)
    return placa, texto, conf


def reconocer_placa(
    frame: np.ndarray, max_variantes: int | None = None
) -> tuple[str, tuple | None]:
    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        return "", None
    placa, _texto, _conf = leer_placa_desde_recorte(recorte)
    return placa, bbox


# ----------------------------------------------------------------
#  PRUEBA CON IMAGEN ESTÁTICA
# ----------------------------------------------------------------
if __name__ == "__main__":
    ruta  = sys.argv[1] if len(sys.argv) > 1 else "prueba_placa.jpg"
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"No se pudo cargar: {ruta}")
    else:
        placa, bbox = reconocer_placa(frame)
        print(f"Placa: {placa!r}")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, placa or "?", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("resultado.jpg", frame)
            print("Guardado en resultado.jpg")
