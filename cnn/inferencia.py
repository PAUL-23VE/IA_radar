"""
cnn/inferencia.py
Reconocimiento de placas basado en YOLOv11 + EasyOCR.

Dado un frame de video:
  1. YOLOv11 (best.pt) detecta la región de la placa
  2. Se recorta y escala el recorte para el OCR
  3. EasyOCR lee el texto de la placa
  4. Se valida/corrige al formato ecuatoriano (ABC-1234)

El modelo `best.pt` fue entrenado con YOLOv11 (1 clase: License_Plate / Placa).
"""

import os
import re
import unicodedata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar")

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------
#  CONFIGURACIÓN
# ----------------------------------------------------------------

# Ruta a los pesos YOLOv11. Se asume ejecución desde la raíz del repo.
RUTA_YOLO = os.getenv("YOLO_MODEL_PATH", "best.pt")

# Confianza mínima para aceptar una detección de placa.
CONF_PLACA = 0.35

# Idioma(s) de EasyOCR.
OCR_LANGS = ["es"]

# OCR solo debe mirar caracteres que aparecen en placas.
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-."


# ----------------------------------------------------------------
#  CARGA DE MODELOS (Singleton — se carga una sola vez)
# ----------------------------------------------------------------

_yolo_cache = None
_ocr_cache = None
_device_cache = None


def resolver_dispositivo() -> str:
    """
    Resuelve el dispositivo de inferencia.

    RADAR_DEVICE=auto  -> CUDA si PyTorch la ve; si no, CPU.
    RADAR_DEVICE=cuda  -> intenta CUDA y cae a CPU si no está disponible.
    RADAR_DEVICE=cpu   -> fuerza CPU.
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    solicitado = os.getenv("RADAR_DEVICE", "auto").strip().lower()

    try:
        import torch

        cuda_disponible = torch.cuda.is_available()
        if solicitado in ("auto", ""):
            dispositivo = "cuda:0" if cuda_disponible else "cpu"
        elif solicitado.startswith("cuda"):
            if cuda_disponible:
                dispositivo = "cuda:0" if solicitado == "cuda" else solicitado
            else:
                print("[GPU] CUDA solicitada, pero no está disponible. Usando CPU.")
                dispositivo = "cpu"
        elif solicitado in ("0", "gpu"):
            dispositivo = "cuda:0" if cuda_disponible else "cpu"
        else:
            dispositivo = "cpu"

        if dispositivo.startswith("cuda"):
            nombre_gpu = torch.cuda.get_device_name(0)
            print(f"[GPU] Usando {dispositivo}: {nombre_gpu}")
        else:
            print("[GPU] Usando CPU.")

    except Exception as exc:
        print(f"[GPU] No se pudo verificar CUDA ({exc}). Usando CPU.")
        dispositivo = "cpu"

    _device_cache = dispositivo
    return dispositivo


def cargar_yolo():
    """Carga (una vez) el detector YOLOv11 de placas."""
    global _yolo_cache
    if _yolo_cache is None:
        if not os.path.exists(RUTA_YOLO):
            raise FileNotFoundError(
                f"No se encontró el modelo YOLO en '{RUTA_YOLO}'. "
                f"Coloca best.pt en la raíz del repo o exporta YOLO_MODEL_PATH."
            )
        print("[YOLO] Cargando detector de placas (best.pt)...")
        _yolo_cache = YOLO(RUTA_YOLO)
        print("[YOLO] Detector listo.")
    return _yolo_cache


def cargar_ocr():
    """Carga (una vez) el lector EasyOCR."""
    global _ocr_cache
    if _ocr_cache is None:
        import easyocr  # import perezoso: arranca lento
        dispositivo = resolver_dispositivo()
        usar_gpu = dispositivo.startswith("cuda")
        print(f"[OCR] Inicializando EasyOCR en {dispositivo}...")
        _ocr_cache = easyocr.Reader(
            OCR_LANGS,
            gpu=dispositivo if usar_gpu else False,
            quantize=not usar_gpu,
            cudnn_benchmark=usar_gpu,
        )
        print("[OCR] EasyOCR listo.")
    return _ocr_cache


# ----------------------------------------------------------------
#  1. DETECTAR REGIÓN DE LA PLACA CON YOLOv11
# ----------------------------------------------------------------

def detectar_region_placa(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detecta la placa de mayor confianza con YOLOv11.

    Devuelve:
      - recorte: imagen BGR de la placa, o None si no hay detección
      - bbox:    (x, y, w, h) en coordenadas del frame, o None
    """
    if frame is None or frame.size == 0:
        return None, None

    modelo = cargar_yolo()
    resultados = modelo(frame, conf=CONF_PLACA, verbose=False, device=resolver_dispositivo())
    cajas = resultados[0].boxes

    if cajas is None or len(cajas) == 0:
        return None, None

    # Nos quedamos con la placa de mayor confianza.
    mejor = int(cajas.conf.argmax())
    x1, y1, x2, y2 = cajas.xyxy[mejor].cpu().numpy().astype(int)

    # Recortar dentro de los límites de la imagen.
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None

    recorte = frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h) — formato esperado por main.py
    return recorte, bbox


# ----------------------------------------------------------------
#  2. PREPROCESAR EL RECORTE PARA EL OCR
# ----------------------------------------------------------------

def _redimensionar_para_ocr(imagen: np.ndarray,
                            alto_min: int = 96,
                            alto_max: int = 180) -> np.ndarray:
    """Escala el recorte a un tamaño más cómodo para EasyOCR."""
    h, _w = imagen.shape[:2]
    if h <= 0:
        return imagen

    escala = 1.0
    if h < alto_min:
        escala = alto_min / h
    elif h > alto_max:
        escala = alto_max / h

    if escala == 1.0:
        return imagen.copy()
    return cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)


def _agregar_borde(imagen: np.ndarray) -> np.ndarray:
    """Añade margen para que OCR no corte caracteres pegados al borde."""
    return cv2.copyMakeBorder(imagen, 12, 12, 18, 18, cv2.BORDER_REPLICATE)


def preprocesar_placa(recorte: np.ndarray) -> np.ndarray | None:
    """
    Mejora el recorte de la placa para el OCR:
    escala de grises → escalado 2x (bicúbico) → umbral de Otsu.
    """
    if recorte is None or recorte.size == 0:
        return None

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def generar_variantes_ocr(recorte: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Genera varias versiones del recorte. EasyOCR a veces lee mejor el color,
    a veces una imagen binaria; probar pocas variantes mejora la robustez.
    """
    if recorte is None or recorte.size == 0:
        return []

    base = _agregar_borde(_redimensionar_para_ocr(recorte))
    gris = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    nitida = cv2.filter2D(
        clahe,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    _, otsu = cv2.threshold(nitida, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptativa = cv2.adaptiveThreshold(
        nitida,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    variantes = [
        ("color", base),
        ("gris_nitida", nitida),
    ]

    # En placas grandes, la franja superior suele decir ECUADOR y confunde al OCR.
    h = base.shape[0]
    if h >= 90:
        inferior = base[int(h * 0.22):, :]
        variantes.append(("zona_numero", inferior))

    variantes.extend([
        ("otsu", otsu),
        ("adaptativa", adaptativa),
    ])

    return variantes


# ----------------------------------------------------------------
#  3. LEER EL TEXTO DE LA PLACA CON EasyOCR
# ----------------------------------------------------------------

def leer_texto_placa(imagen_placa: np.ndarray) -> tuple[str, float]:
    """
    Corre EasyOCR sobre la imagen de la placa.
    Devuelve (texto_crudo, confianza_promedio).
    """
    if imagen_placa is None or imagen_placa.size == 0:
        return "", 0.0

    lector = cargar_ocr()
    detecciones = lector.readtext(
        imagen_placa,
        allowlist=OCR_ALLOWLIST,
        decoder="greedy",
        paragraph=False,
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )

    texto = ""
    suma_conf = 0.0
    n = 0
    for _bbox, fragmento, prob in detecciones:
        texto += fragmento + " "
        suma_conf += prob
        n += 1

    conf_prom = (suma_conf / n) if n > 0 else 0.0
    return texto.strip(), conf_prom


# ----------------------------------------------------------------
#  4. VALIDACIÓN / CORRECCIÓN AL FORMATO ECUATORIANO
# ----------------------------------------------------------------

_LETRAS_DESDE_OCR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "T",
}

_NUMEROS_DESDE_OCR = {
    "A": "4",
    "B": "8",
    "D": "0",
    "E": "3",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "T": "7",
    "Z": "2",
}


def _normalizar_texto_ocr(texto: str) -> str:
    """
    Convierte texto OCR a una forma ASCII manejable, conservando señales útiles.
    La é aparece a menudo cuando EasyOCR confunde un 6.
    """
    reemplazos = {
        "é": "6",
        "É": "6",
        "|": "I",
        "!": "I",
        "¡": "I",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "(": "",
        ")": "",
        "'": "",
        "\"": "",
        "`": "",
        "´": "",
    }
    texto = "".join(reemplazos.get(c, c) for c in texto)
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto.upper()


def _normalizar_zona_letras(chars: list[str]) -> tuple[str, int, int] | None:
    letras = []
    originales = 0
    conversiones = 0

    for c in chars:
        if "A" <= c <= "Z":
            letras.append(c)
            originales += 1
        elif c in _LETRAS_DESDE_OCR:
            letras.append(_LETRAS_DESDE_OCR[c])
            conversiones += 1
        else:
            return None

    if originales < 2:
        return None
    return "".join(letras), originales, conversiones


def _normalizar_zona_numeros(chars: list[str]) -> tuple[str, int, int] | None:
    numeros = []
    originales = 0
    conversiones = 0

    for c in chars:
        if c.isdigit():
            numeros.append(c)
            originales += 1
        elif c in _NUMEROS_DESDE_OCR:
            numeros.append(_NUMEROS_DESDE_OCR[c])
            conversiones += 1
        else:
            return None

    if originales < 2:
        return None
    return "".join(numeros), originales, conversiones


def _crear_candidato(chars: list[str],
                     posiciones: list[int],
                     texto_normalizado: str,
                     penalizacion_extra: float = 0.0) -> tuple[str, float] | None:
    if chars[2].isdigit():
        separador_antes_tercer_char = texto_normalizado[posiciones[1] + 1:posiciones[2]]
        if re.search(r"[\s\-.:]", separador_antes_tercer_char):
            return None

    letras_info = _normalizar_zona_letras(chars[:3])
    numeros_info = _normalizar_zona_numeros(chars[3:])
    if letras_info is None or numeros_info is None:
        return None

    letras, letras_originales, letras_convertidas = letras_info
    numeros, numeros_originales, numeros_convertidos = numeros_info
    placa = f"{letras}-{numeros}"
    if not re.match(r"^[A-Z]{3}-\d{3,4}$", placa):
        return None

    separador = texto_normalizado[posiciones[2] + 1:posiciones[3]]
    bonus_separador = 1.5 if re.search(r"[\s\-.]", separador) else 0.0
    cercania_final = posiciones[0] / max(len(texto_normalizado), 1)
    penalizacion_inicio_medio = 0.0
    if chars[0].isdigit() and posiciones[0] > 0 and texto_normalizado[posiciones[0] - 1].isalnum():
        penalizacion_inicio_medio = 2.5

    score = (
        letras_originales * 2.0
        + numeros_originales * 2.2
        - letras_convertidas * 0.7
        - numeros_convertidos * 1.0
        + len(numeros) * 0.25
        + bonus_separador
        + cercania_final
        - penalizacion_inicio_medio
        - penalizacion_extra
    )
    return placa, score


def extraer_candidatos_placa(texto: str) -> list[tuple[str, float]]:
    """
    Busca placas dentro de texto OCR sucio: por ejemplo
    'ECUAbor EtR-3445' -> ETR-3445.
    """
    normalizado = _normalizar_texto_ocr(texto)
    alfanumericos = [(c, i) for i, c in enumerate(normalizado) if c.isalnum()]
    candidatos: dict[str, float] = {}

    for largo in (7, 6):
        for inicio in range(0, len(alfanumericos) - largo + 1):
            ventana = alfanumericos[inicio:inicio + largo]
            chars = [c for c, _i in ventana]
            posiciones = [i for _c, i in ventana]
            candidato = _crear_candidato(chars, posiciones, normalizado)
            if candidato is None:
                continue
            placa, score = candidato
            candidatos[placa] = max(score, candidatos.get(placa, -999.0))

    # Caso frecuente: OCR mete una letra extra antes del guion: I9IL-2792.
    for inicio in range(0, len(alfanumericos) - 8 + 1):
        ventana = alfanumericos[inicio:inicio + 8]
        chars = [c for c, _i in ventana]
        posiciones = [i for _c, i in ventana]
        zona_letras_larga = chars[:4]
        for indice_a_eliminar in range(4):
            chars_reducidos = chars[:indice_a_eliminar] + chars[indice_a_eliminar + 1:]
            posiciones_reducidas = posiciones[:indice_a_eliminar] + posiciones[indice_a_eliminar + 1:]
            penalizacion = 0.8
            if zona_letras_larga[indice_a_eliminar].isdigit():
                penalizacion += 2.0
            if indice_a_eliminar == 0 and any(c.isdigit() for c in zona_letras_larga):
                penalizacion += 2.0
            candidato = _crear_candidato(
                chars_reducidos,
                posiciones_reducidas,
                normalizado,
                penalizacion_extra=penalizacion,
            )
            if candidato is None:
                continue
            placa, score = candidato
            candidatos[placa] = max(score, candidatos.get(placa, -999.0))

    return sorted(candidatos.items(), key=lambda item: item[1], reverse=True)


def validar_y_corregir_placa(placa_cruda: str) -> str:
    """
    Valida formato ecuatoriano: 3 letras + 3-4 números (ABC-1234).
    Auto-corrige confusiones comunes (O↔0, I↔1, B↔8, Z↔2, S↔5).
    Devuelve la placa formateada o "" si es inválida.
    """
    candidatos = extraer_candidatos_placa(placa_cruda)
    return candidatos[0][0] if candidatos else ""


def leer_placa_desde_recorte(recorte: np.ndarray,
                             max_variantes: int | None = None) -> tuple[str, str, float]:
    """
    Lee el valor de placa desde un recorte YOLO.
    Devuelve: (placa_validada, texto_crudo_elegido, confianza_ocr).
    """
    mejor_placa = ""
    mejor_texto = ""
    mejor_conf = 0.0
    mejor_score = -999.0
    bonus_por_variante = {
        "color": 0.35,
        "zona_numero": 0.25,
        "gris_nitida": 0.0,
        "otsu": -0.15,
        "adaptativa": -1.5,
    }

    variantes = generar_variantes_ocr(recorte)
    if max_variantes is None:
        valor_env = os.getenv("RADAR_OCR_VARIANTS", "").strip()
        if valor_env.isdigit() and int(valor_env) > 0:
            max_variantes = int(valor_env)

    if max_variantes is not None and max_variantes > 0:
        variantes = variantes[:max_variantes]

    for nombre, variante in variantes:
        texto, conf = leer_texto_placa(variante)
        candidatos = extraer_candidatos_placa(texto)
        texto_normalizado = _normalizar_texto_ocr(texto)

        if candidatos:
            placa, score_candidato = candidatos[0]
            bonus = bonus_por_variante.get(nombre, 0.0)
            if "PROVISIONAL" in texto_normalizado and nombre in ("color", "zona_numero"):
                bonus -= 1.0
            score = 100.0 + score_candidato + conf + bonus
        else:
            placa = ""
            score = conf

        if score > mejor_score:
            mejor_placa = placa
            mejor_texto = texto
            mejor_conf = conf
            mejor_score = score

    return mejor_placa, mejor_texto, mejor_conf


# ----------------------------------------------------------------
#  PIPELINE COMPLETO: frame → string placa
# ----------------------------------------------------------------

def reconocer_placa(frame: np.ndarray,
                    max_variantes: int | None = None) -> tuple[str, tuple | None]:
    """
    Función principal. Recibe un frame BGR y devuelve:
      - placa: string reconocido (ej: "ABC-1234") o "" si no detectó / inválida
      - bbox:  coordenadas (x, y, w, h) de la placa en el frame, o None
    """
    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        return "", None

    placa, _texto_crudo, _conf = leer_placa_desde_recorte(recorte, max_variantes=max_variantes)

    return placa, bbox


# ----------------------------------------------------------------
#  PRUEBA con imagen estática
# ----------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ruta = sys.argv[1] if len(sys.argv) > 1 else "prueba_placa.jpg"
    frame = cv2.imread(ruta)

    if frame is None:
        print(f"No se pudo cargar la imagen: {ruta}")
    else:
        placa, bbox = reconocer_placa(frame)
        print(f"Placa detectada: {placa!r}")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, placa or "?", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("resultado.jpg", frame)
            print("Imagen guardada en: resultado.jpg")
