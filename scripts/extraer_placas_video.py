"""
scripts/extraer_placas_video.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extrae recortes de placas CLARAS desde los videos para etiquetar a mano y
reentrenar la CNN con caracteres EC REALES (cierra el domain gap font sintético
→ font real).

Cómo funciona:
  - Recorre los videos indicados.
  - best.pt + ByteTrack rastrea cada placa; por cada track guarda SOLO el mejor
    recorte (mayor área × nitidez) → evita cientos de duplicados de la misma placa.
  - La CNN actual sugiere una lectura para acelerar el etiquetado (solo corriges).

Salida:
  data/datasets/placas_ec_raw/imgs/*.jpg
  data/datasets/placas_ec_raw/etiquetas.csv   (columna placa_real vacía → rellenar)

Uso:
    .venv/bin/python scripts/extraer_placas_video.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cnn"))

from inferencia import leer_placa_cnn  # noqa: E402

# Videos a muestrear (variados para cubrir distintas placas/condiciones).
VIDEOS = [
    "data/uploads/vide1.mp4", "data/uploads/vide3.mp4", "data/uploads/vide4.mp4",
    "data/videos/carros1.mp4", "data/videos/carros2.mp4", "data/videos/carros3.mp4",
    "data/videos/carros4.mp4", "data/videos/carros5.mp4",
]

OUT_DIR = ROOT / "data" / "datasets" / "placas_ec_raw"
IMG_DIR = OUT_DIR / "imgs"
CSV_PATH = OUT_DIR / "etiquetas.csv"

MAX_POR_VIDEO = 25      # tope de placas distintas por video
MIN_PLATE_H = 28        # ignora placas diminutas (ilegibles)
CONF_YOLO = 0.25


def nitidez(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(g, cv2.CV_64F).var()


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(ROOT / "best.pt"))
    try:
        import torch
        if torch.cuda.is_available():
            yolo.to("cuda")
    except Exception:
        pass

    filas = []
    total = 0
    for vid in VIDEOS:
        ruta = ROOT / vid
        if not ruta.exists():
            print(f"[skip] no existe {vid}")
            continue
        cap = cv2.VideoCapture(str(ruta))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        yolo.predictor = None  # reinicia tracker por video
        mejores = {}           # track_id -> (score, crop)
        # Muestrea ~1 de cada 3 frames para velocidad.
        for pos in range(0, nframes, 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break
            res = yolo.track(frame, conf=CONF_YOLO, persist=True,
                             verbose=False, tracker="bytetrack.yaml")[0]
            if res.boxes is None or res.boxes.id is None:
                continue
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                ph = y2 - y1
                if ph < MIN_PLATE_H:
                    continue
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                score = ph * (1.0 + nitidez(crop) / 500.0)  # grande Y nítida
                if tid not in mejores or score > mejores[tid][0]:
                    mejores[tid] = (score, crop.copy())

        vid_tag = Path(vid).stem
        # Ordena por score y toma los mejores N de este video.
        ordenados = sorted(mejores.items(), key=lambda kv: kv[1][0], reverse=True)
        for tid, (score, crop) in ordenados[:MAX_POR_VIDEO]:
            placa_sug, _txt, conf = leer_placa_cnn(crop)
            nombre = f"{vid_tag}_t{tid}.jpg"
            cv2.imwrite(str(IMG_DIR / nombre), crop)
            filas.append({
                "imagen": nombre,
                "placa_sugerida": placa_sug,
                "placa_real": "",          # ← RELLENAR A MANO
                "conf": f"{conf:.2f}",
            })
            total += 1
        cap.release()
        # Escribe el CSV tras CADA video (robusto ante interrupciones).
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["imagen", "placa_sugerida", "placa_real", "conf"])
            w.writeheader()
            w.writerows(filas)
        print(f"[ok] {vid}: {len(ordenados[:MAX_POR_VIDEO])} placas (CSV actualizado)")

    print(f"\n[LISTO] {total} recortes en {IMG_DIR}")
    print(f"[LISTO] CSV para etiquetar: {CSV_PATH}")
    print("\nPaso siguiente: abre el CSV, mira cada imagen en imgs/ y corrige la")
    print("columna placa_real (formato ABC-1234). Borra filas ilegibles.")


if __name__ == "__main__":
    main()
