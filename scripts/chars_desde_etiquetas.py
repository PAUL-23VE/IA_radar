"""
scripts/chars_desde_etiquetas.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Toma las placas etiquetadas a mano (placas_ec_raw/etiquetas.csv con placa_real
rellenada) y genera un dataset de CARACTERES EC REALES para reentrenar la CNN.

Por cada placa:
  - segmenta los caracteres con el MISMO algoritmo de inferencia.py
  - si #segmentos == #caracteres de placa_real → guarda cada char en su clase
  - tolera ±1 (descarta extras por los extremos) por alineación de índice

Salida:
  data/datasets/dataset_chars_ec/<clase>/*.png   (A-Z, 0-9)

Uso:
    .venv/bin/python scripts/chars_desde_etiquetas.py
Luego reentrena (el dataset se suma automáticamente si editas entrenar_ocr_ec.py
para incluirlo, o se mezcla manualmente).
"""
import csv
import re
import sys
import uuid
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cnn"))

from inferencia import _preparar_gris, _segmentar_caracteres  # noqa: E402

RAW = ROOT / "data" / "datasets" / "placas_ec_raw"
CSV_PATH = RAW / "etiquetas.csv"
OUT = ROOT / "data" / "datasets" / "dataset_chars_ec"


def limpiar(placa: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", placa.upper())


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] no existe {CSV_PATH}. Corre extraer_placas_video.py primero.")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    # utf-8-sig: tolera BOM si el CSV se editó/guardó en Excel u otro editor.
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    etiquetadas = [r for r in rows if limpiar(r.get("placa_real", ""))]
    print(f"Placas etiquetadas: {len(etiquetadas)}/{len(rows)}")
    if not etiquetadas:
        print("[ERROR] Ninguna fila tiene placa_real. Rellena el CSV primero.")
        return

    guardados = 0
    exactos = 0
    for r in etiquetadas:
        placa = limpiar(r["placa_real"])
        img_path = RAW / "imgs" / r["imagen"]
        crop = cv2.imread(str(img_path))
        if crop is None:
            continue
        gris = _preparar_gris(crop)
        chars = _segmentar_caracteres(gris)

        # Alineación: necesitamos tantos segmentos como caracteres (±1 tolerado).
        if len(chars) == len(placa):
            pares = list(zip(chars, placa))
            exactos += 1
        elif len(chars) == len(placa) + 1:
            # un extra: descarta el de los extremos más probable (logo/guion)
            pares = list(zip(chars[: len(placa)], placa))
        else:
            continue

        for char_img, etiqueta in pares:
            d = OUT / etiqueta
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"{uuid.uuid4().hex[:10]}.png"), char_img)
            guardados += 1

    print(f"\n[LISTO] {guardados} chars EC reales guardados en {OUT}")
    print(f"        ({exactos} placas segmentaron exacto)")
    print("\nPaso siguiente: reentrenar incluyendo este dataset. Edita")
    print("cnn/entrenar_ocr_ec.py para añadir DATASET_CHARS_EC al ConcatDataset,")
    print("o pídeme que lo integre y lance el reentrenamiento.")


if __name__ == "__main__":
    main()
