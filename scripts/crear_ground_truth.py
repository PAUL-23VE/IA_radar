"""
Crea una plantilla CSV para anotar la placa real de cada imagen.

Uso recomendado:
  python scripts/crear_ground_truth.py \
    --results dev_outputs/dataset_ocr/resultados.csv \
    --out dataset_combinado/ground_truth_test.csv

Luego abre el CSV y llena la columna placa_real.
"""

import argparse
import csv
from pathlib import Path


def leer_resultados(ruta: Path) -> list[dict]:
    with ruta.open("r", newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def normalizar_ruta(ruta: str) -> str:
    path = Path(ruta)
    try:
        return str(path.resolve())
    except OSError:
        return ruta


def main():
    parser = argparse.ArgumentParser(
        description="Crear plantilla para anotar placas reales."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="CSV generado por scripts/probar_dataset.py",
    )
    parser.add_argument(
        "--out",
        default="dataset_combinado/ground_truth_test.csv",
        help="CSV de salida con columna placa_real.",
    )
    args = parser.parse_args()

    resultados_path = Path(args.results)
    salida_path = Path(args.out)
    salida_path.parent.mkdir(parents=True, exist_ok=True)

    filas = leer_resultados(resultados_path)
    campos = [
        "archivo",
        "nombre_archivo",
        "placa_real",
        "placa_predicha",
        "texto_crudo",
        "detectada",
        "confianza_ocr",
        "bbox",
        "notas",
    ]

    with salida_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()

        for fila in filas:
            archivo_imagen = normalizar_ruta(fila.get("archivo", ""))
            writer.writerow({
                "archivo": archivo_imagen,
                "nombre_archivo": Path(archivo_imagen).name,
                "placa_real": "",
                "placa_predicha": fila.get("placa", ""),
                "texto_crudo": fila.get("texto_crudo", ""),
                "detectada": fila.get("detectada", ""),
                "confianza_ocr": fila.get("confianza_ocr", ""),
                "bbox": fila.get("bbox", ""),
                "notas": "",
            })

    print(f"Plantilla creada: {salida_path}")
    print("Llena la columna placa_real y luego ejecuta probar_dataset.py con --ground-truth.")


if __name__ == "__main__":
    main()
