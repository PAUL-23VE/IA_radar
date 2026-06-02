"""
Entorno simple para probar camara, grabar clips y leer placas en vivo.

Ejemplos:
  python scripts/camara_dev.py --source 0
  python scripts/camara_dev.py --source http://192.168.9.129:4747/video
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-radar")

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from inferencia import reconocer_placa  # noqa: E402


def normalizar_source(valor: str):
    return int(valor) if valor.isdigit() else valor


def abrir_writer(frame, carpeta: Path, fps: float):
    carpeta.mkdir(parents=True, exist_ok=True)
    alto, ancho = frame.shape[:2]
    nombre = datetime.now().strftime("clip_%Y%m%d_%H%M%S.mp4")
    ruta = carpeta / nombre
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(ruta), fourcc, fps or 20.0, (ancho, alto))
    return writer, ruta


def main():
    parser = argparse.ArgumentParser(description="Probar camara y OCR de placas.")
    parser.add_argument("--source", default="0", help="Indice de camara, ruta de video o URL HTTP.")
    parser.add_argument("--out", default="dev_outputs/camara")
    parser.add_argument("--every", type=int, default=30, help="Corre OCR cada N frames.")
    parser.add_argument("--variants", type=int, default=2, help="Variantes OCR por lectura.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        help="Dispositivo de inferencia: auto, cpu o cuda.",
    )
    parser.add_argument("--record", action="store_true", help="Empieza grabando desde el arranque.")
    args = parser.parse_args()

    os.environ["RADAR_DEVICE"] = args.device

    source = normalizar_source(args.source)
    salida = Path(args.out).resolve()
    capturas_dir = salida / "capturas"
    clips_dir = salida / "clips"
    capturas_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la fuente de video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    grabando = False
    writer = None
    ruta_clip = None
    ultima_placa = ""
    ultimo_bbox = None
    frame_idx = 0

    print("Controles: q/ESC salir | r grabar/parar | s guardar frame | espacio OCR ahora")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No hay mas frames o se perdio la senal.")
                break

            frame_idx += 1
            correr_ocr = frame_idx % max(args.every, 1) == 0
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(" "):
                correr_ocr = True

            if correr_ocr:
                ultima_placa, ultimo_bbox = reconocer_placa(
                    frame,
                    max_variantes=args.variants,
                )
                if ultima_placa:
                    print(f"[OCR] Placa: {ultima_placa}")

            display = frame.copy()
            if ultimo_bbox:
                x, y, w, h = ultimo_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    ultima_placa or "PLACA",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            estado = "REC" if grabando else "PAUSA"
            cv2.putText(
                display,
                f"{estado} | placa: {ultima_placa or '-'}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if grabando else (255, 255, 255),
                2,
            )
            cv2.imshow("Radar dev", display)

            if grabando and writer is None:
                writer, ruta_clip = abrir_writer(frame, clips_dir, fps)
                print(f"[REC] Grabando en {ruta_clip}")
            if grabando and writer is not None:
                writer.write(frame)

            if args.record and frame_idx == 1:
                grabando = True

            if tecla in (27, ord("q")):
                break
            if tecla == ord("r"):
                grabando = not grabando
                if not grabando and writer is not None:
                    writer.release()
                    print(f"[REC] Clip guardado: {ruta_clip}")
                    writer = None
                    ruta_clip = None
            if tecla == ord("s"):
                nombre = datetime.now().strftime("frame_%Y%m%d_%H%M%S.jpg")
                ruta = capturas_dir / nombre
                cv2.imwrite(str(ruta), frame)
                print(f"[CAPTURA] {ruta}")

    finally:
        if writer is not None:
            writer.release()
            print(f"[REC] Clip guardado: {ruta_clip}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
