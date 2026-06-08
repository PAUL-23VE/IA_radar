"""
scripts/extraer_caracteres.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extrae automáticamente recortes de cada carácter individual de las placas
reales del dataset.

Fuentes de ground-truth (en orden de prioridad):
  1. ground_truth_test.csv  (con placa_real ya rellenada)
  2. EasyOCR sobre train/valid (se usa para autoetiquetar masivamente)

Lógica:
  - Detecta la región de la placa con YOLOv11
  - Segmenta los caracteres con el algoritmo OpenCV de inferencia.py
  - Si #contornos == len(placa_limpia): guarda cada char en su carpeta
  - Fallback: si el segmentador produce ±1 char del esperado, se usa
    alineación por índice descartando los extras por los extremos

Ejecutar desde la raíz del repo:
    .venv/bin/python scripts/extraer_caracteres.py

Salida:
    dataset_propio/   (en la raíz del repo)
        A/  B/  C/ ... Z/  0/  1/ ... 9/
"""

import csv
import re
import sys
import uuid
import os
from pathlib import Path

import cv2
import numpy as np

# Añadir raíz y cnn/ al path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cnn"))

from inferencia import (
    detectar_region_placa,
    segmentar_caracteres,
    cargar_yolo,
)

# ----------------------------------------------------------------
#  Configuración
# ----------------------------------------------------------------
CSV_TEST    = ROOT / "data" / "datasets" / "dataset_combinado" / "ground_truth_test.csv"
SPLITS      = ["train", "valid", "test"]  # carpetas que se recorrerán
DATASET_DIR = ROOT / "data" / "datasets" / "dataset_propio"
CLASES      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ----------------------------------------------------------------
#  Utilidades
# ----------------------------------------------------------------
def limpiar_placa(placa: str) -> str:
    """Devuelve solo los 6-7 alphanum de la placa, sin guión."""
    p = re.sub(r"[^A-Z0-9]", "", placa.upper())
    return p if 6 <= len(p) <= 7 else ""


def guardar_char(imagen: np.ndarray, clase: str, destino: Path) -> None:
    clase_dir = destino / clase
    clase_dir.mkdir(parents=True, exist_ok=True)
    nombre = clase_dir / f"{uuid.uuid4().hex[:12]}.png"
    cv2.imwrite(str(nombre), imagen)


# ----------------------------------------------------------------
#  Fuente 1: CSV de test con placa_real ya rellenada
# ----------------------------------------------------------------
def cargar_csv(ruta_csv: Path) -> dict[str, str]:
    """Devuelve {nombre_archivo: placa_limpia}."""
    mapa = {}
    if not ruta_csv.exists():
        return mapa
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            placa = limpiar_placa(fila.get("placa_real", ""))
            nombre = fila.get("nombre_archivo", "").strip()
            if placa and nombre:
                mapa[nombre] = placa
    return mapa

# ----------------------------------------------------------------
#  Fuente 2: EasyOCR para autoetiquetar train/valid
# ----------------------------------------------------------------
_easyocr_reader = None

def ocr_placa(recorte: np.ndarray) -> str:
    """Lee la placa con EasyOCR y devuelve la forma limpia."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[OCR] Iniciando EasyOCR (solo una vez)...")
        _easyocr_reader = easyocr.Reader(["es"], gpu=True)
        print("[OCR] Listo")
    import unicodedata
    detecciones = _easyocr_reader.readtext(
        recorte,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.",
        paragraph=False,
    )
    texto = " ".join(t for _, t, _ in detecciones)
    # Buscar patrón de placa ecuatoriana
    texto = unicodedata.normalize("NFD", texto.upper())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    m = re.search(r"([A-Z]{3})[\s\-.]?(\d{3,4})", texto)
    if m:
        return m.group(1) + m.group(2)
    return ""

# ----------------------------------------------------------------
#  Alineación tolerante ±1
# ----------------------------------------------------------------
def alinear_chars(segmentados: list, placa: str) -> list[tuple] | None:
    """Si los contornos difieren en ±1, recorta extremos para alinear."""
    n_seg = len(segmentados)
    n_plc = len(placa)
    if n_seg == n_plc:
        return list(zip(segmentados, placa))
    # Si hay un char extra al inicio o al final, probamos ambas opciones
    if n_seg == n_plc + 1:
        # Descartar el primero
        if all(s == p for s, p in zip(placa, placa)):  # siempre True, solo por estructura
            for drop_start in (True, False):
                subset = segmentados[1:] if drop_start else segmentados[:-1]
                return list(zip(subset, placa))
    return None


# ----------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("  EXTRACCIÓN DE CARACTERES REALES")
    print("=" * 60)

    DATASET_DIR.mkdir(exist_ok=True)

    # Mapa de ground-truth del CSV de test
    gt_csv = cargar_csv(CSV_TEST)
    print(f"[CSV] {len(gt_csv)} placas con ground-truth en el CSV")

    # Contadores globales
    conteo   = {c: 0 for c in CLASES}
    total_ok = 0
    total_ko = 0
    total_ocr_ok = 0

    for split in SPLITS:
        img_dir = ROOT / "data" / "datasets" / "dataset_combinado" / split / "images"
        if not img_dir.exists():
            print(f"[SKIP] {split}: directorio no encontrado")
            continue

        imagenes = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"\n[{split.upper()}] {len(imagenes)} imágenes")

        for idx, ruta_img in enumerate(imagenes, 1):
            frame = cv2.imread(str(ruta_img))
            if frame is None:
                continue

            if idx % 100 == 0:
                print(f"  → {idx}/{len(imagenes)} procesadas...")

            # ── 1. Obtener ground truth ──────────────────────────
            nombre = ruta_img.name
            placa  = gt_csv.get(nombre, "")

            # ── 2. Detectar y recortar la región de la placa ────
            recorte, _ = detectar_region_placa(frame)
            if recorte is None:
                total_ko += 1
                continue

            # ── 3. Si no hay placa en CSV, usar EasyOCR ──────────
            if not placa and split in ("train", "valid"):
                placa = ocr_placa(recorte)
                if placa:
                    total_ocr_ok += 1

            if not placa:
                total_ko += 1
                continue

            # ── 4. Segmentar caracteres ──────────────────────────
            chars = segmentar_caracteres(recorte)

            # Alinear: exacto o ±1
            pares = alinear_chars(chars, placa)
            if pares is None:
                total_ko += 1
                continue

            # ── 5. Guardar cada carácter ─────────────────────────
            for img_char, clase in pares:
                if clase not in CLASES:
                    continue
                guardar_char(img_char, clase, DATASET_DIR)
                conteo[clase] += 1

            total_ok += 1

    # ── Resumen ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Placas extraídas:      {total_ok}")
    print(f"  Placas omitidas:       {total_ko}")
    print(f"  Total caracteres:      {sum(conteo.values())}")
    print("=" * 60)
    print("\nDistribución por clase:")
    for c in CLASES:
        barra = "█" * min(conteo[c], 50)
        print(f"  {c}: {conteo[c]:>5}  {barra}")

    print(f"\nDataset guardado en: {DATASET_DIR}")


if __name__ == "__main__":
    main()
