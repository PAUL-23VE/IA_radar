"""
main.py
Pipeline completo del sistema de reconocimiento de placas.

Flujo:
  1. Captura frame desde DroidCam (WiFi) — hilo principal
  2. Hilo de reconocimiento (RecognitionWorker) detecta placa en paralelo
  3. Consulta PostgreSQL → datos del vehículo y propietario
  4. Mide velocidad con dos líneas virtuales + lógica difusa
  5. Si hay multa → registra en BD

Ejecutar: python main.py [demo|live|laptop]
"""

import queue
import sys
import os
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "database"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "velocidad"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from inferencia    import reconocer_placa
from db_connection import buscar_vehiculo, imprimir_vehiculo, registrar_multa
from logica_difusa import clasificar_velocidad
from camara        import URL_STREAM

DISTANCIA_REFERENCIA_METROS = 5.0

ESTADO_VELOCIDAD = 0
ESTADO_PLACA     = 1
ESTADO_BD        = 2


# ----------------------------------------------------------------
#  Hilo de reconocimiento de placas (no bloquea el loop principal)
# ----------------------------------------------------------------

class RecognitionWorker(threading.Thread):
    """
    Procesa frames en un hilo separado para no bloquear la captura.
    Solo mantiene el frame más reciente en la cola (descarta frames viejos).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._inbox  = queue.Queue(maxsize=1)
        self._result = ("", None, 0.0)
        self._lock   = threading.Lock()
        self._active = True

    def submit(self, frame: np.ndarray) -> None:
        """Envía un frame para procesar. Descarta el anterior si aún no fue consumido."""
        try:
            self._inbox.get_nowait()  # vaciar si hay uno pendiente
        except queue.Empty:
            pass
        try:
            self._inbox.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_result(self) -> tuple[str, tuple | None, float]:
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._active = False

    def run(self) -> None:
        while self._active:
            try:
                frame = self._inbox.get(timeout=0.1)
                placa, bbox, conf = reconocer_placa(frame)
                with self._lock:
                    self._result = (placa, bbox, conf)
            except queue.Empty:
                continue


# ----------------------------------------------------------------
#  Apertura de cámara (DroidCam / local)
# ----------------------------------------------------------------

def _abrir_camara(cam_url) -> cv2.VideoCapture | None:
    if isinstance(cam_url, str) and cam_url.startswith("http"):
        base = cam_url.rstrip("/")
        if base.endswith("/video"):
            base = base[:-6]
        rutas = [f"{base}/video", f"{base}/", f"{base}/mjpeg", f"{base}/live"]
        print("[Sistema] Buscando señal de video…")
        for ruta in rutas:
            cap = cv2.VideoCapture(ruta)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[Sistema] Cámara conectada: {ruta}")
                    return cap
            cap.release()
        return None
    cap = cv2.VideoCapture(cam_url)
    return cap if cap.isOpened() else None


# ----------------------------------------------------------------
#  Pipeline principal
# ----------------------------------------------------------------

def procesar_vehiculo(cam_url=URL_STREAM, distancia_m: float = DISTANCIA_REFERENCIA_METROS):
    print("\n" + "=" * 60)
    print("  SISTEMA INTEGRADO DE PLACAS — UTA")
    print("=" * 60)

    cap = _abrir_camara(cam_url)
    if cap is None:
        print("[ERROR] No se pudo abrir la cámara.")
        print("Asegúrate de que DroidCam esté activo en la misma red WiFi.")
        return

    # Leer primer frame para conocer dimensiones
    ret, frame_init = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el primer frame.")
        cap.release()
        return

    alto, ancho = frame_init.shape[:2]
    linea_a_y   = int(alto * 0.30)
    linea_b_y   = int(alto * 0.70)

    # Estado del pipeline
    estado          = ESTADO_VELOCIDAD
    fgbg            = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    t_a = t_b       = 0.0
    cruzó_linea_a   = False
    cruzó_linea_b   = False
    velocidad_kmh   = 0.0
    resultado_difuso = None
    placa_detectada  = ""
    datos_vehiculo   = None

    # Iniciar hilo de reconocimiento
    worker = RecognitionWorker()
    worker.start()

    print("[Sistema] Iniciando monitoreo… (ESC = salir, R = reiniciar)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_display = frame.copy()

            # ─── MÓDULO VELOCIDAD ──────────────────────────────────
            if estado == ESTADO_VELOCIDAD:
                mascara      = fgbg.apply(frame)
                _, mask_bin  = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
                kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask_clean   = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                mask_clean   = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                vehiculo_y = None
                for cnt in contornos:
                    if cv2.contourArea(cnt) > 3000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        vehiculo_y = y + h
                        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        break

                cv2.line(frame_display, (0, linea_a_y), (ancho, linea_a_y), (255, 0, 0), 2)
                cv2.putText(frame_display, "Linea A", (10, linea_a_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.line(frame_display, (0, linea_b_y), (ancho, linea_b_y), (0, 0, 255), 2)
                cv2.putText(frame_display, "Linea B", (10, linea_b_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame_display, "MIDIENDO VELOCIDAD…", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if vehiculo_y is not None:
                    if not cruzó_linea_a and vehiculo_y > linea_a_y:
                        t_a = time.time()
                        cruzó_linea_a = True
                        print("[Velocidad] Cruzó Línea A")
                    if cruzó_linea_a and not cruzó_linea_b and vehiculo_y > linea_b_y:
                        t_b = time.time()
                        cruzó_linea_b = True
                        dt = t_b - t_a
                        if dt > 0:
                            velocidad_kmh = round((distancia_m / dt) * 3.6, 2)
                        resultado_difuso = clasificar_velocidad(velocidad_kmh)
                        estado = ESTADO_PLACA
                        print(f"[Velocidad] {velocidad_kmh} km/h → {resultado_difuso['clasificacion']}")

            # ─── MÓDULO PLACA (hilo worker) ────────────────────────
            elif estado == ESTADO_PLACA:
                cv2.putText(frame_display,
                            f"V: {velocidad_kmh} km/h  [{resultado_difuso['clasificacion'].upper()}]",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame_display, "BUSCANDO PLACA…",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Enviar frame al worker (no bloquea)
                worker.submit(frame)

                # Leer último resultado del worker
                placa, bbox, conf = worker.get_result()
                if placa and conf >= 0.50:
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    placa_detectada = placa
                    estado = ESTADO_BD
                    print(f"[Placa] Reconocida: {placa}  (conf={conf:.2f})")

                # Timeout 5s para no quedarse bloqueado
                if t_b > 0 and time.time() - t_b > 5.0:
                    print("[Sistema] Timeout buscando placa. Reiniciando…")
                    estado = ESTADO_VELOCIDAD
                    cruzó_linea_a = cruzó_linea_b = False
                    t_a = t_b = 0.0

            # ─── MÓDULO DB + HUD FINAL ─────────────────────────────
            elif estado == ESTADO_BD:
                if datos_vehiculo is None:
                    datos_vehiculo = buscar_vehiculo(placa_detectada)
                    if resultado_difuso["clasificacion"] in ("normal", "multa"):
                        registrar_multa(
                            placa_detectada, velocidad_kmh,
                            resultado_difuso["clasificacion"],
                            resultado_difuso["dias_sin_ingreso"],
                        )

                cv2.rectangle(frame_display, (10, 10), (500, 170), (0, 0, 0), -1)
                cv2.putText(frame_display, f"PLACA: {placa_detectada}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_display, f"VELOCIDAD: {velocidad_kmh} km/h",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                color_d = (0, 255, 0)
                if resultado_difuso["clasificacion"] == "multa":
                    color_d = (0, 0, 255)
                elif resultado_difuso["clasificacion"] == "normal":
                    color_d = (0, 255, 255)

                cv2.putText(frame_display,
                            f"ESTADO: {resultado_difuso['clasificacion'].upper()}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_d, 2)

                if datos_vehiculo:
                    cv2.putText(frame_display,
                                f"{datos_vehiculo['marca']} {datos_vehiculo['modelo']}",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame_display, "R=reiniciar  ESC=salir",
                        (10, alto - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Sistema Integrado de Placas", frame_display)

            # waitKey(1) para máximo throughput — no limitar a 33fps
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            elif tecla in (ord("r"), ord("R")):
                estado = ESTADO_VELOCIDAD
                cruzó_linea_a = cruzó_linea_b = False
                t_a = t_b = 0.0
                placa_detectada = ""
                datos_vehiculo  = None
                resultado_difuso = None
                velocidad_kmh   = 0.0
                # Limpiar resultado del worker
                with worker._lock:
                    worker._result = ("", None, 0.0)
                print("[Sistema] Reiniciando…")

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------
#  Modo demo (sin cámara)
# ----------------------------------------------------------------

def demo_sin_camara(placa_prueba: str = "ABC-1234", velocidad_prueba: float = 35.0):
    print("\n" + "=" * 60)
    print("  MODO DEMO — sin cámara ni CNN")
    print("=" * 60)

    resultado = clasificar_velocidad(velocidad_prueba)
    print(f"\n[Demo] Velocidad: {velocidad_prueba} km/h")
    print(resultado["mensaje"])
    print(f"  Membresía: {resultado['grados_membresia']}")

    print(f"\n[Demo] Placa: {placa_prueba}")
    datos = buscar_vehiculo(placa_prueba)
    imprimir_vehiculo(datos)

    if resultado["clasificacion"] in ("normal", "multa") and datos:
        registrar_multa(placa_prueba, velocidad_prueba,
                        resultado["clasificacion"],
                        resultado["dias_sin_ingreso"])
        print("✅ Multa registrada.")


# ----------------------------------------------------------------
#  Entrada
# ----------------------------------------------------------------

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if modo == "demo":
        demo_sin_camara("ABC-1234", 35.0)
        demo_sin_camara("XYZ-4567", 18.0)
        demo_sin_camara("KLM-1234", 25.0)
    elif modo == "live":
        procesar_vehiculo(cam_url=URL_STREAM)
    elif modo == "laptop":
        procesar_vehiculo(cam_url=0)
