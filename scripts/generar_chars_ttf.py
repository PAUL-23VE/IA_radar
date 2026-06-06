"""
scripts/generar_chars_ttf.py
Genera caracteres sintéticos 0-9 A-Z con fuentes TTF reales que coinciden con
la tipografía de las placas ecuatorianas (DejaVu Sans Bold ≈ font EC).

Por qué TTF y no Hershey de OpenCV:
  El generador anterior (generar_sinteticos_placa.py) usaba fuentes vectoriales
  de OpenCV (FONT_HERSHEY_*) que NO se parecen a la tipografía real de placa,
  así que el clasificador entrenado con ellas fallaba en placas reales
  (G→H, T→3…). DejaVu/Liberation Bold renderizadas con PIL son casi idénticas
  a los glifos reales (verificado contra GTR-3445), cerrando el gap de dominio.

Augmentación que imita captura de tránsito:
  rotación, perspectiva, blur de movimiento, ruido, brillo/contraste,
  erosión/dilatación (grosor de trazo), recorte/traslación y CLAHE — el mismo
  preprocesamiento que aplica la inferencia, para que train e inferencia
  vean la misma distribución.

Salida: dataset_propio_ttf/<CLASE>/<nombre>.png  (escala de grises)

Uso:
    .venv/bin/python scripts/generar_chars_ttf.py --por-clase 600
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT    = Path(__file__).resolve().parents[1]
DESTINO = ROOT / "data" / "datasets" / "dataset_propio_ttf"
CLASES  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Fuentes que se parecen a la placa EC (bold/heavy condensed sans).
# FiraSansCompressed-Heavy y FiraSansCondensed-Heavy son las más cercanas al
# trazo grueso y proporciones condensadas de la tipografía ANT Ecuador.
CANDIDATAS = [
    # Primarias: condensadas/heavy — más cercanas a la fuente EC real
    "/usr/share/fonts/TTF/FiraSansCompressed-Heavy.ttf",
    "/usr/share/fonts/TTF/FiraSansCondensed-Heavy.ttf",
    "/usr/share/fonts/TTF/FiraSansCompressed-ExtraBold.ttf",
    "/usr/share/fonts/TTF/FiraSansCondensed-ExtraBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/gsfonts/NimbusSansNarrow-Bold.otf",
    # Secundarias: variedad de trazo para generalización
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf",
    "/usr/share/fonts/carlito/Carlito-Bold.ttf",
]


def fuentes_disponibles() -> list[str]:
    fs = [f for f in CANDIDATAS if os.path.exists(f)]
    if not fs:
        raise SystemExit("[ERROR] No se encontró ninguna fuente TTF candidata.")
    return fs


def render_glifo(ch: str, font_path: str, lienzo: int = 96) -> np.ndarray:
    """Renderiza un carácter negro centrado sobre fondo blanco (gris uint8)."""
    f = ImageFont.truetype(font_path, int(lienzo * 0.72))
    img = Image.new("L", (lienzo, lienzo), 255)
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), ch, font=f)
    w, h = r - l, b - t
    d.text(((lienzo - w) / 2 - l, (lienzo - h) / 2 - t), ch, font=f, fill=0)
    return np.array(img)


# ----------------------------------------------------------------
#  Augmentación
# ----------------------------------------------------------------

def _perspectiva(img: np.ndarray, mag: float) -> np.ndarray:
    h, w = img.shape
    d = mag * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.random.uniform(-d, d, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)


def _grosor(img: np.ndarray) -> np.ndarray:
    # Sesgar hacia trazos más gruesos (plaça EC usa trazo muy bold)
    k = random.choice([1, 2, 2, 3, 3])
    op = random.choice(["dilate", "dilate", "erode", "none"])
    if op == "none":
        return img
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # texto es oscuro -> erosionar el fondo blanco = engrosar texto
    if op == "dilate":
        return cv2.erode(img, ker)
    return cv2.dilate(img, ker)


def augmentar(glifo: np.ndarray, rng: random.Random) -> np.ndarray:
    img = glifo.copy()

    # grosor de trazo
    img = _grosor(img)

    # rotación ±9°
    ang = rng.uniform(-9, 9)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # perspectiva leve
    if rng.random() < 0.6:
        img = _perspectiva(img, rng.uniform(0.02, 0.10))

    # traslación / recorte: pega el glifo en un lienzo más grande desplazado
    pad = rng.randint(2, 14)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    dx, dy = rng.randint(-6, 6), rng.randint(-6, 6)
    Mt = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, Mt, (img.shape[1], img.shape[0]), borderValue=255)

    # blur de movimiento / desenfoque
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        if rng.random() < 0.5:
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k          # blur horizontal (movimiento)
            img = cv2.filter2D(img, -1, ker)
        else:
            img = cv2.GaussianBlur(img, (k, k), 0)

    # brillo / contraste
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-40, 40)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # CLAHE (igual que la inferencia)
    if rng.random() < 0.5:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

    # ruido gaussiano
    if rng.random() < 0.5:
        noise = rng.uniform(3, 18)
        img = np.clip(img.astype(np.float32) + np.random.randn(*img.shape) * noise, 0, 255).astype(np.uint8)

    # polaridad invertida ocasional (texto claro sobre fondo oscuro)
    if rng.random() < 0.15:
        img = 255 - img

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-clase", type=int, default=3000)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    fuentes = fuentes_disponibles()
    print(f"[TTF] Fuentes: {len(fuentes)}")
    for f in fuentes:
        print("   ", f)

    rng = random.Random(args.semilla)
    np.random.seed(args.semilla)

    # Pre-render de glifos base por (clase, fuente)
    base = {ch: [render_glifo(ch, f) for f in fuentes] for ch in CLASES}

    total = 0
    for ch in CLASES:
        nombre_dir = ch  # ImageFolder usa el nombre de carpeta como etiqueta
        out = DESTINO / nombre_dir
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.por_clase):
            glifo = rng.choice(base[ch])
            aug = augmentar(glifo, rng)
            cv2.imwrite(str(out / f"{ch}_{i:04d}.png"), aug)
            total += 1
        print(f"   {ch}: {args.por_clase}")

    print(f"\n[OK] {total} imágenes en {DESTINO}")


if __name__ == "__main__":
    main()
