#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cámara REAL del celular (Android) por USB en HD -> webcam virtual /dev/video9
# Usa scrcpy (espejo de la cámara) + v4l2loopback. Baja latencia + 1080p.
#
# Requisitos (ya instalados en esta máquina): adb, scrcpy >=2.0, v4l2loopback.
# El celular debe ser Android 12+ (espejo de cámara de scrcpy).
#
# Uso:
#   scripts/camara_usb.sh                 # cámara trasera, 1920x1080
#   scripts/camara_usb.sh front 1280x720  # frontal, 720p
#
# Luego en el radar (http://localhost:8000/app/):
#   Tipo de Entrada = "URL / Stream"  ->  escribe  9  ->  Iniciar Radar
#   (9 = /dev/video9, la webcam virtual creada aquí)
# ---------------------------------------------------------------------------
set -e

FACING="${1:-back}"     # back | front
SIZE="${2:-1920x1080}"  # ver tamaños disponibles: scrcpy --list-camera-sizes
DEV=/dev/video9

# 1) ¿celular conectado y autorizado?
if ! adb get-state >/dev/null 2>&1; then
  echo "[!] No se detecta el celular por USB."
  echo "    - Activa 'Depuración USB' en Opciones de desarrollador."
  echo "    - Conéctalo por USB y acepta el diálogo 'Permitir depuración USB'."
  echo "    - 'adb devices' debe listarlo como 'device' (no 'unauthorized')."
  adb devices
  exit 1
fi

# 2) webcam virtual /dev/video9 (carga el módulo solo si falta)
if [ ! -e "$DEV" ]; then
  echo "[*] Cargando v4l2loopback en $DEV (pide sudo)…"
  sudo modprobe v4l2loopback video_nr=9 card_label="Celular USB" exclusive_caps=1
fi

# 3) espejo de la cámara del celular -> /dev/video9
echo "[*] scrcpy: cámara $FACING $SIZE -> $DEV   (Ctrl+C para detener)"
exec scrcpy \
  --video-source=camera \
  --camera-facing="$FACING" \
  --camera-size="$SIZE" \
  --no-audio \
  --no-window \
  --v4l2-sink="$DEV"
