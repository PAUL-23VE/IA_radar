"""
Prueba deteccion de placa + OCR sobre imagenes de dataset_combinado.

Ejemplos:
  python scripts/probar_dataset.py --split test --limit 20
  python scripts/probar_dataset.py --images dataset_combinado/valid/images --show
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar")

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from inferencia import (  # noqa: E402
    detectar_region_placa,
    leer_placa_desde_recorte,
)

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalizar_placa(valor: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (valor or "").upper())


def iterar_imagenes(carpeta: Path, limite: int | None):
    rutas = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if limite is not None:
        rutas = rutas[:limite]
    return rutas


def dibujar_resultado(frame, bbox, placa, texto_crudo, confianza):
    salida = frame.copy()
    etiqueta = placa or texto_crudo or "SIN OCR"
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            salida,
            f"{etiqueta} ({confianza:.2f})",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            salida,
            "SIN DETECCION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
    return salida


def cargar_ground_truth(ruta: str | None) -> dict[str, str]:
    if not ruta:
        return {}

    path = Path(ruta)
    if not path.exists():
        raise SystemExit(f"No existe el archivo ground truth: {path}")

    ground_truth = {}
    with path.open("r", newline="", encoding="utf-8") as archivo:
        reader = csv.DictReader(archivo)
        if "placa_real" not in (reader.fieldnames or []):
            raise SystemExit("El CSV ground truth debe tener una columna placa_real.")

        for fila in reader:
            placa_real = normalizar_placa(fila.get("placa_real", ""))
            if not placa_real:
                continue

            archivo_fila = fila.get("archivo", "")
            nombre_fila = fila.get("nombre_archivo", "")
            if archivo_fila:
                ground_truth[str(Path(archivo_fila).resolve())] = placa_real
                ground_truth[Path(archivo_fila).name] = placa_real
            if nombre_fila:
                ground_truth[nombre_fila] = placa_real

    return ground_truth


def procesar_imagen(ruta: Path, max_variantes: int | None = None):
    frame = cv2.imread(str(ruta))
    if frame is None:
        return {
            "archivo": str(ruta),
            "detectada": "no",
            "texto_crudo": "",
            "placa": "",
            "confianza_ocr": "0.000",
            "bbox": "",
            "error": "No se pudo leer la imagen",
        }, None

    recorte, bbox = detectar_region_placa(frame)
    if recorte is None:
        return {
            "archivo": str(ruta),
            "detectada": "no",
            "texto_crudo": "",
            "placa": "",
            "confianza_ocr": "0.000",
            "bbox": "",
            "error": "",
        }, dibujar_resultado(frame, None, "", "", 0.0)

    placa, texto_crudo, confianza = leer_placa_desde_recorte(
        recorte,
        max_variantes=max_variantes,
    )

    return {
        "archivo": str(ruta),
        "detectada": "si",
        "texto_crudo": texto_crudo,
        "placa": placa,
        "confianza_ocr": f"{confianza:.3f}",
        "bbox": ",".join(str(v) for v in bbox),
        "error": "",
    }, dibujar_resultado(frame, bbox, placa, texto_crudo, confianza)


def main():
    parser = argparse.ArgumentParser(
        description="Probar best.pt + EasyOCR contra imagenes locales."
    )
    parser.add_argument("--dataset", default="data/datasets/dataset_combinado")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--images", help="Carpeta de imagenes; reemplaza --dataset/--split.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", default="outputs/dev_outputs/dataset_ocr")
    parser.add_argument(
        "--ground-truth",
        help="CSV con columnas archivo/nombre_archivo y placa_real.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        help="Dispositivo de inferencia: auto, cpu o cuda.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=0,
        help="Cantidad de variantes OCR a probar. 0 usa todas.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.environ["RADAR_DEVICE"] = args.device

    carpeta = Path(args.images) if args.images else Path(args.dataset) / args.split / "images"
    carpeta = carpeta.resolve()
    salida = Path(args.out).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    imagenes = iterar_imagenes(carpeta, args.limit)
    if not imagenes:
        raise SystemExit(f"No encontre imagenes en {carpeta}")

    ground_truth = cargar_ground_truth(args.ground_truth)
    csv_path = salida / "resultados.csv"
    campos = [
        "archivo",
        "detectada",
        "texto_crudo",
        "placa",
        "placa_real",
        "ocr_correcto",
        "confianza_ocr",
        "bbox",
        "error",
    ]
    total_detectadas = 0
    total_placas = 0
    total_con_gt = 0
    total_correctas = 0

    with csv_path.open("w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.DictWriter(archivo_csv, fieldnames=campos)
        writer.writeheader()

        for idx, ruta in enumerate(imagenes, start=1):
            fila, imagen_salida = procesar_imagen(
                ruta,
                max_variantes=args.variants if args.variants > 0 else None,
            )

            placa_real = (
                ground_truth.get(str(ruta.resolve()))
                or ground_truth.get(ruta.name)
                or ""
            )
            fila["placa_real"] = placa_real
            if placa_real:
                total_con_gt += 1
                es_correcta = normalizar_placa(fila["placa"]) == placa_real
                fila["ocr_correcto"] = "si" if es_correcta else "no"
                if es_correcta:
                    total_correctas += 1
            else:
                fila["ocr_correcto"] = ""

            writer.writerow(fila)

            if fila["detectada"] == "si":
                total_detectadas += 1
            if fila["placa"]:
                total_placas += 1

            if imagen_salida is not None:
                destino = salida / f"{idx:04d}_{ruta.name}"
                cv2.imwrite(str(destino), imagen_salida)

                if args.show:
                    cv2.imshow("Resultado dataset", imagen_salida)
                    if cv2.waitKey(0) & 0xFF == 27:
                        break

            print(
                f"[{idx}/{len(imagenes)}] det={fila['detectada']} "
                f"placa={fila['placa'] or '-'} "
                f"real={fila['placa_real'] or '-'} "
                f"ok={fila['ocr_correcto'] or '-'} "
                f"raw={fila['texto_crudo'] or '-'}"
            )

    if args.show:
        cv2.destroyAllWindows()

    print("\nResumen")
    print(f"  Imagenes procesadas : {len(imagenes)}")
    porcentaje_detectadas = (total_detectadas / len(imagenes)) * 100 if imagenes else 0.0
    porcentaje_validas_total = (total_placas / len(imagenes)) * 100 if imagenes else 0.0
    porcentaje_validas_detectadas = (
        (total_placas / total_detectadas) * 100 if total_detectadas else 0.0
    )
    print(f"  Con placa detectada : {total_detectadas} ({porcentaje_detectadas:.1f}%)")
    print(f"  Con texto validado  : {total_placas} ({porcentaje_validas_total:.1f}%)")
    print(f"  Validas/detectadas  : {porcentaje_validas_detectadas:.1f}%")
    if ground_truth:
        porcentaje_gt = (total_correctas / total_con_gt) * 100 if total_con_gt else 0.0
        print(f"  Con placa real      : {total_con_gt}")
        print(f"  OCR exacto          : {total_correctas} ({porcentaje_gt:.1f}%)")
    print(f"  CSV                 : {csv_path}")
    print(f"  Imagenes anotadas   : {salida}")


if __name__ == "__main__":
    main()
