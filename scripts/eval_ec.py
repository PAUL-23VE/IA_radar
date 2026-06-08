"""
scripts/eval_ec.py
Evalua el pipeline OCR contra el set de validacion EC hecho a mano
(dataset_combinado/val_ec.csv). Reporta acc por caracter y por placa.

Uso: .venv/bin/python scripts/eval_ec.py
"""
import csv, os, sys
import cv2
sys.path.insert(0, "cnn")
import inferencia as I

CSV = "data/datasets/dataset_combinado/val_ec.csv"
IMGDIR = "data/datasets/dataset_combinado/test/images"


def char_acc(pred, real):
    pred = pred.replace("-", ""); real = real.replace("-", "")
    n = max(len(pred), len(real))
    if n == 0:
        return 0.0
    ok = sum(1 for a, b in zip(pred, real) if a == b)
    return ok / len(real) if real else 0.0


def main():
    I.cargar_cnn()
    rows = list(csv.DictReader(open(CSV)))
    pl_ok = 0
    ch_sum = 0.0
    for r in rows:
        p = os.path.join(IMGDIR, r["archivo"])
        frame = cv2.imread(p)
        rec, _ = I.detectar_region_placa(frame)
        if rec is None:
            print(f"  {r['placa']:10s} -> (sin deteccion)")
            continue
        placa, raw, conf = I.leer_placa_cnn(rec)
        real = r["placa"]
        pred = placa or raw
        ca = char_acc(pred, real)
        ch_sum += ca
        exact = (placa == real)
        pl_ok += int(exact)
        mark = "OK " if exact else "   "
        print(f"  {mark}{real:10s} pred={pred:12s} raw={raw:12s} char={ca:.2f} conf={conf:.2f}")
    n = len(rows)
    print(f"\n  Placas exactas: {pl_ok}/{n} = {pl_ok/n:.1%}")
    print(f"  Char-acc medio: {ch_sum/n:.1%}")


if __name__ == "__main__":
    main()
