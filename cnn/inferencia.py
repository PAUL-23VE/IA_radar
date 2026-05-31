"""
cnn/inferencia.py
Dado un frame de video:
  1. Detecta la región de la placa (RegionProps, OpenCV)
  2. Segmenta cada carácter individualmente (RegionProps)
  3. Predice cada carácter con la CNN entrenada
  4. Reconstruye el string de la placa
"""

import cv2
import numpy as np
import tensorflow as tf
import re
from skimage.measure import label, regionprops
from modelo import CLASES, TAMANO_IMAGEN

RUTA_MODELO = "models/modelo_entrenado.h5"


# ----------------------------------------------------------------
#  CARGAR MODELO (se hace una sola vez al importar)
# ----------------------------------------------------------------

_modelo_cache = None

def cargar_modelo():
    global _modelo_cache
    if _modelo_cache is None:
        print("[CNN] Cargando modelo entrenado...")
        _modelo_cache = tf.keras.models.load_model(RUTA_MODELO)
        print("[CNN] Modelo listo.")
    return _modelo_cache


# ----------------------------------------------------------------
#  1. DETECTAR REGIÓN DE LA PLACA EN EL FRAME
# ----------------------------------------------------------------

def ordenar_puntos(pts):
    """Ordena los 4 puntos del rectángulo para la corrección de perspectiva."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left
    return rect

def correccion_perspectiva(imagen, pts):
    """Aplica warpPerspective para corregir inclinación."""
    rect = ordenar_puntos(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(imagen, M, (maxWidth, maxHeight))

def detectar_region_placa(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detecta la región de la placa usando skimage.measure.regionprops
    y aplica corrección de perspectiva e inclinación.
    """
    # 1. Escala de grises y Gaussian Blur (reducción de ruido)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # 2. Canny Edge Detection
    bordes = cv2.Canny(blur, 50, 150)

    # 3. Operación morfológica (Closing) para conectar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

    # 4. Connected Components y RegionProps
    label_img = label(bordes_cerrados > 0)
    propiedades = regionprops(label_img)

    mejor_prop = None
    mejor_area = 0

    for prop in propiedades:
        minr, minc, maxr, maxc = prop.bbox
        w = maxc - minc
        h = maxr - minr
        if h == 0: continue
        
        aspect_ratio = w / float(h)
        area = prop.area

        # Análisis geométrico: Aspect ratio y área
        if 2.0 < aspect_ratio < 6.0 and area > 1000 and area > mejor_area:
            mejor_prop = prop
            mejor_area = area

    if mejor_prop is not None:
        minr, minc, maxr, maxc = mejor_prop.bbox
        w = maxc - minc
        h = maxr - minr
        
        # Recuperar la máscara de la mejor región para hallar los 4 vértices (inclinación)
        mascara = np.zeros_like(bordes_cerrados, dtype=np.uint8)
        for coord in mejor_prop.coords:
            mascara[coord[0], coord[1]] = 255
            
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            cnt = max(contornos, key=cv2.contourArea)
            # bounding box rotado para corrección de perspectiva
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Recortar y corregir perspectiva automáticamente
            recorte_warped = correccion_perspectiva(frame, box)
            
            # El bounding box clásico para dibujar en pantalla
            bbox_pantalla = (minc, minr, w, h)
            return recorte_warped, bbox_pantalla

    return None, None


# ----------------------------------------------------------------
#  2. SEGMENTAR CARACTERES DENTRO DEL RECORTE DE PLACA
# ----------------------------------------------------------------

def segmentar_caracteres(imagen_placa: np.ndarray) -> list[np.ndarray]:
    """
    Recibe el recorte de la placa y devuelve una lista de imágenes,
    una por carácter, ordenadas de izquierda a derecha.
    Segmenta utilizando skimage.measure.regionprops.
    """
    # 1. Escala de grises
    if len(imagen_placa.shape) == 3:
        gris = cv2.cvtColor(imagen_placa, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_placa.copy()

    # Redimensionar para mejor segmentación
    gris = cv2.resize(gris, (400, 120))

    # 2. Binarización adaptativa (thresholding)
    binaria = cv2.adaptiveThreshold(
        gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    # 3. Operaciones morfológicas (Opening/Closing) para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    # 4. Etiquetado (Connected Components) y RegionProps
    label_img = label(binaria > 0)
    propiedades = regionprops(label_img)

    caracteres = []
    for prop in propiedades:
        minr, minc, maxr, maxc = prop.bbox
        w = maxc - minc
        h = maxr - minr
        area = prop.area

        # Análisis geométrico de cada letra
        if 8 < w < 80 and 30 < h < 110 and h > w and area > 100:
            char_img = binaria[minr:maxr, minc:maxc]

            # Añadir padding para no cortar los bordes
            char_img = cv2.copyMakeBorder(
                char_img, 6, 6, 6, 6,
                cv2.BORDER_CONSTANT, value=0
            )

            # Resize al tamaño que espera la CNN
            char_img = cv2.resize(char_img, TAMANO_IMAGEN)
            char_img = char_img.astype('float32') / 255.0

            # Guardar con su posición minc (X) para ordenar después
            caracteres.append((minc, char_img))

    # Ordenar de izquierda a derecha
    caracteres.sort(key=lambda c: c[0])

    return [img for _, img in caracteres]


# ----------------------------------------------------------------
#  3. PREDECIR CADA CARÁCTER CON LA CNN
# ----------------------------------------------------------------

def predecir_caracteres(lista_chars: list[np.ndarray]) -> str:
    """
    Recibe la lista de imágenes de caracteres (32×32, float)
    y devuelve el string de la placa reconstruida.
    """
    if not lista_chars:
        return ""

    modelo = cargar_modelo()

    # Stack de todos los chars en un batch
    batch = np.array(lista_chars).reshape(-1, 32, 32, 1)
    predicciones = modelo.predict(batch, verbose=0)

    placa = ""
    for pred in predicciones:
        idx = np.argmax(pred)
        confianza = pred[idx]

        # Solo aceptar si la confianza supera el umbral
        if confianza > 0.60:
            placa += CLASES[idx]

    return placa


# ----------------------------------------------------------------
#  HEURÍSTICAS DE VALIDACIÓN
# ----------------------------------------------------------------

def validar_y_corregir_placa(placa_cruda: str) -> str:
    """
    Usa expresiones regulares y heurísticas para validar que la placa 
    tenga el formato Ecuatoriano: 3 letras y 3-4 números.
    Auto-corrige confusiones comunes (ej. O por 0, I por 1).
    Devuelve la placa formateada o un string vacío si es inválida.
    """
    if len(placa_cruda) < 6 or len(placa_cruda) > 7:
        return ""

    letras = placa_cruda[:3]
    numeros = placa_cruda[3:]

    # Corregir letras comunes confundidas con números
    letras = letras.replace('0', 'O').replace('1', 'I').replace('8', 'B')
    # Corregir números comunes confundidos con letras
    numeros = numeros.replace('O', '0').replace('I', '1').replace('B', '8')
    numeros = numeros.replace('Z', '2').replace('S', '5')

    placa_corregida = letras + "-" + numeros

    # Validar con RegEx: 3 letras de la A-Z, seguido de un guión, seguido de 3 o 4 dígitos.
    if re.match(r'^[A-Z]{3}-\d{3,4}$', placa_corregida):
        return placa_corregida
    else:
        return ""


# ----------------------------------------------------------------
#  PIPELINE COMPLETO: frame → string placa
# ----------------------------------------------------------------

def reconocer_placa(frame: np.ndarray) -> tuple[str, tuple | None]:
    """
    Función principal. Recibe un frame BGR y devuelve:
      - placa: string reconocido (ej: "ABC-1234") o "" si no detectó o es inválida
      - bbox:  coordenadas (x, y, w, h) de la placa en el frame, o None
    """
    recorte, bbox = detectar_region_placa(frame)

    if recorte is None:
        return "", None

    chars = segmentar_caracteres(recorte)
    placa_cruda = predecir_caracteres(chars)

    placa_validada = validar_y_corregir_placa(placa_cruda)

    return placa_validada, bbox


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
        print(f"Placa detectada: {placa}")
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, placa, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imwrite("resultado.jpg", frame)
            print("Imagen guardada en: resultado.jpg")
