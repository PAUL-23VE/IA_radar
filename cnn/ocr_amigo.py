"""
cnn/ocr_amigo.py
OCR del repo del amigo (Plate_Detection_Segmentation_OCR_sin_dependencias) portado
a onnxruntime para correr en Python 3.14 (sin TensorFlow).

Cadena: placa BGR -> deskew (Hough) -> U-Net segmenta caracteres -> CNN clasifica
-> formato placa Ecuador (3 letras + 3/4 digitos).

Generaliza mucho mejor que nuestra segmentacion por componentes conexos porque la
segmentacion de caracteres es APRENDIDA (U-Net), no umbrales fijos.

Modelos ONNX (convertidos 1 vez desde .keras con scripts/convert_keras_onnx.py):
    models/amigo_seg_unet.onnx   (entrada 96x256x1, salida mascara 96x256)
    models/amigo_ocr_cnn.onnx    (entrada 64x64x1,  salida 36 clases softmax)

El preprocesado/postprocesado es copia FIEL del repo del amigo para reproducir su
exactitud (mask_to_boxes, refinar tinta, prepare_crop, leer_placa por subsecuencia).
"""
import os
import threading
from itertools import combinations

import cv2
import numpy as np
import onnxruntime as ort

_DIR = os.path.dirname(os.path.abspath(__file__))
_RUTA_SEG = os.path.join(_DIR, "..", "models", "amigo_seg_unet.onnx")
_RUTA_CNN = os.path.join(_DIR, "..", "models", "amigo_ocr_cnn.onnx")
_RUTA_CLASSES = os.path.join(
    _DIR, "..", "Plate_Detection_Segmentation_OCR_sin_dependencias",
    "ml", "models", "ocr", "Modelos", "classes.txt")

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHAR_ASPECT = 0.60

# parametros de mask_to_boxes (espejo del cfg del amigo)
_THRESHOLD = 0.50
_MIN_AREA_RATIO = 0.002
_PADDING = 0.08

_lock = threading.Lock()
_seg = None
_cnn = None


# ----------------------------------------------------------------
#  Carga de sesiones ONNX (singleton thread-safe)
# ----------------------------------------------------------------
def _cargar():
    global _seg, _cnn, CLASSES
    if _seg is None or _cnn is None:
        with _lock:
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(6, os.cpu_count() or 4)
            if _seg is None:
                _seg = ort.InferenceSession(os.path.abspath(_RUTA_SEG),
                                            sess_options=so,
                                            providers=["CPUExecutionProvider"])
            if _cnn is None:
                _cnn = ort.InferenceSession(os.path.abspath(_RUTA_CNN),
                                            sess_options=so,
                                            providers=["CPUExecutionProvider"])
            if os.path.exists(_RUTA_CLASSES):
                with open(_RUTA_CLASSES, encoding="utf-8") as f:
                    CLASSES = [ln.strip() for ln in f if ln.strip()]
    return _seg, _cnn


def disponible() -> bool:
    return os.path.exists(_RUTA_SEG) and os.path.exists(_RUTA_CNN)


# ----------------------------------------------------------------
#  Deskew por rotacion (Hough) — copia del amigo
# ----------------------------------------------------------------
def _estimar_angulo(gris):
    edges = cv2.Canny(gris, 50, 150, apertureSize=3)
    h, w = gris.shape[:2]
    segmentos = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=max(30, w // 6),
        minLineLength=max(20, w // 4), maxLineGap=max(5, w // 20))
    if segmentos is None:
        return None
    angulos = []
    for x1, y1, x2, y2 in segmentos[:, 0]:
        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if deg < -45:
            deg += 90
        elif deg > 45:
            deg -= 90
        if abs(deg) <= 45:
            angulos.append(deg)
    return float(np.median(angulos)) if angulos else None


def _rotar(img, angulo):
    h, w = img.shape[:2]
    centro = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - centro[0]
    M[1, 2] += nh / 2.0 - centro[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _deskew(crop, umbral=1.5, maxg=30.0):
    if crop is None or crop.size == 0:
        return crop
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    a = _estimar_angulo(gris)
    if a is None or abs(a) < umbral or abs(a) > maxg:
        return crop
    return _rotar(crop, a)


# ----------------------------------------------------------------
#  Segmentacion: U-Net -> mascara -> cajas — copia del amigo
# ----------------------------------------------------------------
def _split_box_by_projection(binary, x, y, w, h, expected_w):
    n = max(1, int(round(w / expected_w)))
    if n <= 1:
        return [(x, y, w, h)]
    column_sum = binary[y:y + h, x:x + w].sum(axis=0).astype("float32")
    smooth = max(1, int(expected_w * 0.15))
    if smooth > 1:
        column_sum = np.convolve(column_sum, np.ones(smooth) / smooth, mode="same")
    window = max(1, int(expected_w * 0.40))
    min_seg = max(2, int(expected_w * 0.35))
    cuts = []
    prev = 0
    for i in range(1, n):
        center = i * w / n
        lo = max(prev + min_seg, int(center - window))
        hi = min(w - min_seg, int(center + window))
        cut = int(center) if hi <= lo else lo + int(np.argmin(column_sum[lo:hi]))
        cut = max(prev + min_seg, min(cut, w - min_seg))
        if prev < cut < w:
            cuts.append(cut)
            prev = cut
    bounds = [0] + cuts + [w]
    sub = []
    for i in range(len(bounds) - 1):
        sw = bounds[i + 1] - bounds[i]
        if sw > 0:
            sub.append((x + bounds[i], y, sw, h))
    return sub


def _mask_to_boxes(mask, original_shape, threshold=_THRESHOLD,
                   min_area_ratio=_MIN_AREA_RATIO, padding=_PADDING,
                   char_aspect=CHAR_ASPECT, height_keep_ratio=0.50):
    original_h, original_w = original_shape[:2]
    mask_h, mask_w = mask.shape[:2]
    binary = (mask >= threshold).astype("uint8") * 255
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = mask_h * mask_w * min_area_ratio
    raw = [cv2.boundingRect(c) for c in contours
           if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] >= min_area]
    if not raw:
        return [], binary
    max_h = max(h for (_, _, _, h) in raw)
    chars = [b for b in raw if b[3] >= height_keep_ratio * max_h] or raw
    median_h = float(np.median([h for (_, _, _, h) in chars]))
    median_w = float(np.median([w for (_, _, w, _) in chars]))
    if len(chars) >= 3:
        expected_w = max(median_w, median_h * char_aspect * 0.8)
    else:
        expected_w = max(1.0, median_h * char_aspect)
    split = []
    for (x, y, w, h) in chars:
        if w >= 1.5 * expected_w:
            split.extend(_split_box_by_projection(binary, x, y, w, h, expected_w))
        else:
            split.append((x, y, w, h))
    boxes = []
    sx = original_w / mask_w
    sy = original_h / mask_h
    for (x, y, w, h) in split:
        x1, y1 = int(x * sx), int(y * sy)
        x2, y2 = int((x + w) * sx), int((y + h) * sy)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(original_w, x2 + px); y2 = min(original_h, y2 + py)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return sorted(boxes, key=lambda b: b[0]), binary


def _refinar_cajas_por_tinta(gris, cajas, tol_alto=1.7):
    if not cajas:
        return cajas
    H, W = gris.shape[:2]
    blur = cv2.GaussianBlur(gris, (3, 3), 0)
    _, tinta = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    tinta = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num, _l, stats, _c = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    alto_med = float(np.median([max(1, y2 - y1) for (x1, y1, x2, y2) in cajas]))
    centros = [(x1 + x2) / 2.0 for (x1, y1, x2, y2) in cajas]
    nuevas = []
    for i, (x1, y1, x2, y2) in enumerate(cajas):
        lim_izq = 0 if i == 0 else int((centros[i - 1] + centros[i]) / 2)
        lim_der = W if i == len(cajas) - 1 else int((centros[i] + centros[i + 1]) / 2)
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
        ancho_caja = max(1, x2 - x1)
        for c in range(1, num):
            cx, cy, cw, ch, _a = stats[c]
            if ch > tol_alto * alto_med or ch < 0.40 * alto_med:
                continue
            if cw > 1.8 * ancho_caja and cw > tol_alto * alto_med:
                continue
            ox1, ox2 = max(x1, cx), min(x2, cx + cw)
            if ox2 <= ox1 or (ox2 - ox1) / float(cw) < 0.50:
                continue
            nx1 = max(lim_izq, min(nx1, cx)); ny1 = min(ny1, cy)
            nx2 = min(lim_der, max(nx2, cx + cw)); ny2 = max(ny2, cy + ch)
        nuevas.append((max(0, nx1), max(0, ny1), min(W, nx2), min(H, ny2)))
    return nuevas


def _filtrar_ruido(cajas, crops, alto_min_rel=0.55, aspecto_max=1.4, minimo=4):
    if not cajas or len(cajas) < minimo:
        return cajas, crops
    alturas = sorted(max(1, y2 - y1) for (x1, y1, x2, y2) in cajas)
    umbral = alto_min_rel * alturas[len(alturas) // 2]
    elegidas = []
    for i, (x1, y1, x2, y2) in enumerate(cajas):
        alto = max(1, y2 - y1); ancho = max(1, x2 - x1)
        if alto < umbral or ancho / alto > aspecto_max:
            continue
        elegidas.append(i)
    if not elegidas:
        return cajas, crops
    return [cajas[i] for i in elegidas], [crops[i] for i in elegidas]


def _segmentar(plate_bgr):
    seg, _ = _cargar()
    gris = plate_bgr if plate_bgr.ndim == 2 else cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # entrada U-Net: 96x256x1 (h=96, w=256), float32 0-255
    entrada = cv2.resize(gris, (256, 96), interpolation=cv2.INTER_AREA)
    entrada = entrada.astype("float32")[None, :, :, None]
    nombre_in = seg.get_inputs()[0].name
    mascara = seg.run(None, {nombre_in: entrada})[0][0, :, :, 0]
    cajas, _ = _mask_to_boxes(mascara, plate_bgr.shape)
    cajas = _refinar_cajas_por_tinta(gris, cajas)
    crops = [plate_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in cajas]
    cajas, crops = _filtrar_ruido(cajas, crops)
    return crops


# ----------------------------------------------------------------
#  OCR: prepare_crop + leer_placa por subsecuencia — copia del amigo
# ----------------------------------------------------------------
def _prepare_crop(gray_crop, target_h=64, target_w=64):
    arr = np.array(gray_crop)
    mask = arr < 140
    if mask.any():
        ys, xs = np.where(mask)
        x1 = max(0, xs.min() - 3); y1 = max(0, ys.min() - 3)
        x2 = min(arr.shape[1], xs.max() + 4); y2 = min(arr.shape[0], ys.max() + 4)
        gray_crop = gray_crop[y1:y2, x1:x2]
    h, w = gray_crop.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h)) * 0.88
    new_w = max(1, min(target_w, int(w * scale)))
    new_h = max(1, min(target_h, int(h * scale)))
    resized_char = cv2.resize(gray_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    light = gray_crop[gray_crop > 120]
    bg = int(np.clip(int(np.median(light)) if light.size else 245, 210, 255))
    out = np.full((target_h, target_w), bg, dtype=np.uint8)
    x = (target_w - new_w) // 2; y = (target_h - new_h) // 2
    out[y:y + new_h, x:x + new_w] = resized_char
    return out.astype("float32")


_MAX_CANDIDATOS = 12


def _leer_placa(crops, num_letras=3, digitos_validos=(4, 3)):
    if not crops:
        return "", []
    _, cnn = _cargar()
    letras_ids = [i for i, c in enumerate(CLASSES) if c.isalpha()]
    digitos_ids = [i for i, c in enumerate(CLASSES) if c.isdigit()]
    validos = [c for c in crops if c is not None and c.size > 0]
    if not validos:
        return "", []
    batch = np.stack([
        _prepare_crop(c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))[:, :, None]
        for c in validos]).astype("float32")
    nombre_in = cnn.get_inputs()[0].name
    preds = cnn.run(None, {nombre_in: batch})[0]
    candidatos = []
    for crop, pred in zip(validos, preds):
        il = max(letras_ids, key=lambda i: pred[i])
        idg = max(digitos_ids, key=lambda i: pred[i])
        candidatos.append({
            "letra": CLASSES[il], "conf_letra": float(pred[il]),
            "digito": CLASSES[idg], "conf_digito": float(pred[idg]),
            "alto": int(crop.shape[0]),
        })
    n = len(candidatos)
    if n > _MAX_CANDIDATOS:
        candidatos = sorted(candidatos,
                            key=lambda c: max(c["conf_letra"], c["conf_digito"]),
                            reverse=True)[:_MAX_CANDIDATOS]
        n = len(candidatos)
    alturas = sorted(c["alto"] for c in candidatos)
    alto_medio = alturas[len(alturas) // 2]

    def _emitir(indices):
        texto, confs = "", []
        for pos, i in enumerate(indices):
            c = candidatos[i]
            if pos < num_letras:
                texto += c["letra"]; confs.append(c["conf_letra"])
            else:
                texto += c["digito"]; confs.append(c["conf_digito"])
        return texto, confs

    mejor = None
    for nd in digitos_validos:
        largo = num_letras + nd
        if n < largo:
            continue
        for indices in combinations(range(n), largo):
            score = sum(candidatos[i]["conf_letra"] if pos < num_letras
                        else candidatos[i]["conf_digito"]
                        for pos, i in enumerate(indices))
            penal = sum(max(0.0, (alto_medio * 0.6 - candidatos[i]["alto"]) / alto_medio)
                        for i in indices)
            saltos = (indices[-1] - indices[0] + 1) - largo
            s = score / largo - 0.12 * saltos - 0.35 * penal
            if nd == 4:
                s += 0.05
            if mejor is None or s > mejor[0]:
                mejor = (s, indices)
    indices = mejor[1] if mejor is not None else tuple(range(n))
    return _emitir(indices)


# ----------------------------------------------------------------
#  Interfaz publica
# ----------------------------------------------------------------
MIN_CHARS = 5
MAX_CHARS = 7


def leer_placa_amigo(plate_bgr) -> tuple[str, str, float]:
    """
    Placa recortada (BGR) -> (placa_formateada 'ABC-1234', texto_crudo, confianza).
    Devuelve ('', '', 0.0) si la segmentacion no produce un numero de chars valido.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "", "", 0.0
    plate = _deskew(plate_bgr)
    crops = _segmentar(plate)
    if not (MIN_CHARS <= len(crops) <= MAX_CHARS):
        return "", "", 0.0
    texto, confs = _leer_placa(crops)
    conf = float(np.mean(confs)) if confs else 0.0
    placa = ""
    if len(texto) in (6, 7):
        cand = f"{texto[:3]}-{texto[3:]}"
        import re
        if re.match(r"^[A-Z]{3}-\d{3,4}$", cand):
            placa = cand
    return placa, texto, conf
