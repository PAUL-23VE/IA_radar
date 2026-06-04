"""
main.py
Pipeline completo del sistema de reconocimiento de placas.

Flujo:
  1. Captura frame desde la fuente seleccionada — hilo principal
  2. Hilo de reconocimiento (RecognitionWorker): YOLO + CNN de caracteres
  3. Mide velocidad con dos líneas virtuales + lógica difusa (sanción en horas)
  4. Registra el evento (placa, velocidad, multa, captura) en registros/eventos.json

Modos de ejecución:
  python main.py demo                      # sin cámara, datos ficticios
  python main.py video <ruta>             # archivo de video (mp4, avi…)
  python main.py camara [indice]          # cámara física (default: 0)
  python main.py digital [url_o_indice]  # DroidCam/scrcpy (default: URL_STREAM)

  Alias legacy: laptop → camara 0 | live → digital URL_STREAM

Controles en ventana:
  F        → pantalla completa / ventana normal
  Arrastrar línea A/B con mouse → reposicionar umbrales de velocidad
  R        → reiniciar pipeline
  ESC      → salir
"""

import queue
import sys
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "velocidad"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from inferencia    import reconocer_placa
from logica_difusa import clasificar_velocidad
from registro      import registrar_evento
from camara        import URL_STREAM

DISTANCIA_REFERENCIA_METROS = 5.0

ESTADO_VELOCIDAD = 0
ESTADO_PLACA     = 1
ESTADO_REGISTRO  = 2

WIN_NAME     = "Sistema Integrado de Placas"
DIR_CAPTURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")


# ----------------------------------------------------------------
#  Hilo de reconocimiento de placas (no bloquea el loop principal)
# ----------------------------------------------------------------

class RecognitionWorker(threading.Thread):
    """
    Procesa frames en un hilo separado para no bloquear la captura.
    Solo mantiene el frame más reciente en la cola (descarta frames viejos).
    Cada lectura completada se acumula para que el loop principal pueda votarlas.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._inbox   = queue.Queue(maxsize=1)
        self._result  = ("", None, 0.0)   # última lectura (para el bbox/HUD)
        self._pendientes = []             # lecturas nuevas no consumidas
        self._lock    = threading.Lock()
        self._active  = True

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._inbox.get_nowait()
        except queue.Empty:
            pass
        try:
            self._inbox.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_result(self) -> tuple[str, tuple | None, float]:
        with self._lock:
            return self._result

    def drenar_lecturas(self) -> list[tuple[str, tuple | None, float]]:
        """Devuelve y limpia las lecturas nuevas desde la última llamada."""
        with self._lock:
            nuevas, self._pendientes = self._pendientes, []
            return nuevas

    def reset_lecturas(self) -> None:
        with self._lock:
            self._pendientes = []
            self._result = ("", None, 0.0)

    def stop(self) -> None:
        self._active = False

    def run(self) -> None:
        while self._active:
            try:
                frame = self._inbox.get(timeout=0.1)
                placa, bbox, conf = reconocer_placa(frame)
                with self._lock:
                    self._result = (placa, bbox, conf)
                    self._pendientes.append((placa, bbox, conf))
            except queue.Empty:
                continue


# ----------------------------------------------------------------
#  Votación temporal de placas (multi-frame)
# ----------------------------------------------------------------

class VotadorPlaca:
    """
    Acumula lecturas válidas de varios fotogramas y produce un consenso por
    votación posición-a-posición. En un sistema en tiempo real la misma placa
    se ve en muchos frames; votar entre ellos corrige los errores de un solo
    frame (blur, ángulo) y lleva la precisión cerca del 100%.
    """

    def __init__(self, min_votos: int = 4, conf_min: float = 0.45):
        self.min_votos = min_votos
        self.conf_min  = conf_min
        self._lecturas: list[str] = []

    def agregar(self, placa: str, conf: float) -> None:
        if placa and conf >= self.conf_min:
            self._lecturas.append(placa)

    @property
    def n(self) -> int:
        return len(self._lecturas)

    def consenso(self) -> str:
        """Consenso si hay suficientes votos; '' en caso contrario."""
        from collections import Counter
        if len(self._lecturas) < self.min_votos:
            return ""
        # Agrupar por longitud (ABC-NNN vs ABC-NNNN) y usar la más frecuente
        longitud = Counter(len(p) for p in self._lecturas).most_common(1)[0][0]
        grupo    = [p for p in self._lecturas if len(p) == longitud]
        # Mayoría por posición
        return "".join(
            Counter(p[i] for p in grupo).most_common(1)[0][0]
            for i in range(longitud)
        )

    def reset(self) -> None:
        self._lecturas = []


# ----------------------------------------------------------------
#  Líneas de velocidad con endpoints libremente arrastrables
# ----------------------------------------------------------------

R_ENDPOINT = 12   # radio de agarre de un endpoint (px)
R_LINEA    = 10   # distancia máxima al segmento para agarrar la línea entera


def _dist_punto_segmento(px, py, x1, y1, x2, y2) -> float:
    """Distancia del punto (px,py) al segmento (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return ((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2) ** 0.5


def _lado_linea(px, py, x1, y1, x2, y2) -> float:
    """Signo del producto cruzado: positivo/negativo según el lado de la línea."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _callback_mouse(event, x, y, flags, param):
    """
    Click+drag sobre un endpoint (círculo) → mueve solo ese punto.
    Click+drag sobre el cuerpo de la línea → desplaza toda la línea.
    Sin restricciones de posición: las líneas van a donde el usuario quiera.
    """
    st = param   # dict con linea_a, linea_b, drag

    if event == cv2.EVENT_LBUTTONDOWN:
        st["drag"] = None
        for nombre in ("a", "b"):
            ln = st[f"linea_{nombre}"]
            # ¿click cerca de endpoint 1?
            if ((x - ln["x1"]) ** 2 + (y - ln["y1"]) ** 2) ** 0.5 <= R_ENDPOINT:
                st["drag"] = (nombre, "p1"); break
            # ¿click cerca de endpoint 2?
            if ((x - ln["x2"]) ** 2 + (y - ln["y2"]) ** 2) ** 0.5 <= R_ENDPOINT:
                st["drag"] = (nombre, "p2"); break
            # ¿click cerca del cuerpo de la línea?
            if _dist_punto_segmento(x, y, ln["x1"], ln["y1"], ln["x2"], ln["y2"]) <= R_LINEA:
                st["drag"] = (nombre, "linea")
                st["drag_ox"] = x; st["drag_oy"] = y
                st["drag_lx1"] = ln["x1"]; st["drag_ly1"] = ln["y1"]
                st["drag_lx2"] = ln["x2"]; st["drag_ly2"] = ln["y2"]
                break

    elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
        nombre, parte = st["drag"]
        ln = st[f"linea_{nombre}"]
        if parte == "p1":
            ln["x1"], ln["y1"] = x, y
        elif parte == "p2":
            ln["x2"], ln["y2"] = x, y
        else:  # mover línea completa
            dx = x - st["drag_ox"]; dy = y - st["drag_oy"]
            ln["x1"] = st["drag_lx1"] + dx; ln["y1"] = st["drag_ly1"] + dy
            ln["x2"] = st["drag_lx2"] + dx; ln["y2"] = st["drag_ly2"] + dy

    elif event == cv2.EVENT_LBUTTONUP:
        st["drag"] = None


# ----------------------------------------------------------------
#  Guardar captura al detectar placa
# ----------------------------------------------------------------

def _guardar_captura(frame: np.ndarray, placa: str, velocidad: float,
                     clasificacion: str, horas: int) -> str:
    os.makedirs(DIR_CAPTURAS, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre    = f"{placa.replace('-', '')}_{ts}.jpg"
    ruta      = os.path.join(DIR_CAPTURAS, nombre)

    img = frame.copy()
    h, w = img.shape[:2]

    # Fondo semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    color_estado = (0, 0, 255) if clasificacion == "multa" else \
                   (0, 255, 255) if clasificacion == "normal" else (0, 255, 0)

    cv2.putText(img, f"PLACA: {placa}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, f"VELOCIDAD: {velocidad:.1f} km/h",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    etiqueta = clasificacion.upper()
    if clasificacion == "multa":
        etiqueta += f"  ({horas}h indisponible)"
    cv2.putText(img, f"ESTADO: {etiqueta}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

    cv2.putText(img, ts, (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imwrite(ruta, img)
    return ruta


# ----------------------------------------------------------------
#  Apertura de cámara
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
        print("Verifica la fuente de video o que DroidCam esté activo.")
        return

    ret, frame_init = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el primer frame.")
        cap.release()
        return

    alto, ancho = frame_init.shape[:2]

    # Detectar si la fuente es un archivo de video (no cámara en vivo).
    # Para archivos: usar los timestamps del propio video para calcular velocidad
    # (time.time() mediría tiempo de CPU, no tiempo del video → velocidades imposibles).
    # También limitar el display al FPS real del video para que no vaya rapidísimo.
    es_archivo = isinstance(cam_url, str) and not cam_url.startswith("http")
    fps_video   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms    = max(1, int(1000 / fps_video)) if es_archivo else 1

    # Líneas con endpoints libremente arrastrables (sin restricciones de posición).
    # Inicialmente horizontales; el usuario arrastra endpoints o el cuerpo completo.
    estado_lineas = {
        "linea_a": {"x1": 0,     "y1": int(alto * 0.30),
                    "x2": ancho, "y2": int(alto * 0.30)},
        "linea_b": {"x1": 0,     "y1": int(alto * 0.70),
                    "x2": ancho, "y2": int(alto * 0.70)},
        "drag": None,
    }

    # Ventana redimensionable; arranca en pantalla completa
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN_NAME, _callback_mouse, estado_lineas)

    pantalla_completa = True

    estado           = ESTADO_VELOCIDAD
    fgbg             = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    t_a = t_b        = 0.0
    cruzó_linea_a    = False
    cruzó_linea_b    = False
    velocidad_kmh    = 0.0
    resultado_difuso = None
    placa_detectada  = ""
    evento_registrado = False
    captura_guardada = False

    worker  = RecognitionWorker()
    votador = VotadorPlaca()
    worker.start()

    print("[Sistema] Iniciando monitoreo…")
    print("  F=pantalla completa  |  Arrastrar líneas A/B con mouse")
    print("  R=reiniciar  |  ESC=salir")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Leer geometría actual de las líneas (actualizada por el mouse)
            la = estado_lineas["linea_a"]
            lb = estado_lineas["linea_b"]

            frame_display = frame.copy()

            # ─── MÓDULO VELOCIDAD ──────────────────────────────────
            if estado == ESTADO_VELOCIDAD:
                mascara      = fgbg.apply(frame)
                _, mask_bin  = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
                kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask_clean   = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                mask_clean   = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Tomar el contorno más grande como vehículo principal
                veh_cx = veh_cy = None
                mejor_area = 0
                for cnt in contornos:
                    area = cv2.contourArea(cnt)
                    if area > 1500 and area > mejor_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        veh_cx = bx + bw // 2   # centro horizontal
                        veh_cy = by + bh         # borde inferior
                        mejor_area = area
                        cv2.rectangle(frame_display, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

                # Dibujar líneas con endpoints visibles (círculos)
                drag_nombre = estado_lineas["drag"][0] if estado_lineas["drag"] else None
                for nombre, ln, color_base in [("a", la, (255, 100, 0)), ("b", lb, (0, 0, 255))]:
                    color = (0, 180, 255) if drag_nombre == nombre else color_base
                    p1 = (ln["x1"], ln["y1"]); p2 = (ln["x2"], ln["y2"])
                    cv2.line(frame_display, p1, p2, color, 2)
                    cv2.circle(frame_display, p1, R_ENDPOINT, color, -1)
                    cv2.circle(frame_display, p2, R_ENDPOINT, color, -1)
                    etiq = f"Linea {nombre.upper()}  [arrastrar]"
                    cv2.putText(frame_display, etiq,
                                (ln["x1"] + 8, ln["y1"] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
                cv2.putText(frame_display, "MIDIENDO VELOCIDAD…", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if veh_cx is not None and veh_cy is not None:
                    # Detección de cruce por cambio de signo del producto cruzado.
                    # _lado_linea devuelve un valor cuyo signo indica a qué lado
                    # de la línea se encuentra el punto; si cambia entre frames
                    # el vehículo cruzó la línea.
                    lado_a = _lado_linea(veh_cx, veh_cy,
                                         la["x1"], la["y1"], la["x2"], la["y2"])
                    lado_b = _lado_linea(veh_cx, veh_cy,
                                         lb["x1"], lb["y1"], lb["x2"], lb["y2"])

                    if not cruzó_linea_a and lado_a >= 0:
                        t_a = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                        cruzó_linea_a = True
                        print("[Velocidad] Cruzó Línea A")
                    if cruzó_linea_a and not cruzó_linea_b and lado_b >= 0:
                        t_b = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
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

                # Solo enviar al worker si hay movimiento real en el frame.
                # Evita leer placas de vehículos estáticos o frames vacíos.
                mascara_mov   = fgbg.apply(frame)
                _, mov_bin    = cv2.threshold(mascara_mov, 200, 255, cv2.THRESH_BINARY)
                pixeles_mov   = cv2.countNonZero(mov_bin)
                hay_movimiento = pixeles_mov > 800   # umbral: ~0.1% de un frame 1080p

                if hay_movimiento:
                    worker.submit(frame)

                # Acumular todas las lecturas nuevas para votación temporal
                for p, bb, c in worker.drenar_lecturas():
                    votador.agregar(p, c)
                _, bbox, _ = worker.get_result()
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_display, f"Lecturas: {votador.n}/{votador.min_votos}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                # Consenso multi-frame → placa definitiva
                placa_consenso = votador.consenso()
                if placa_consenso:
                    placa_detectada = placa_consenso
                    estado = ESTADO_REGISTRO
                    captura_guardada = False
                    print(f"[Placa] Consenso ({votador.n} lecturas): {placa_detectada}")
                    votador.reset()

                # Timeout: 5 s reales (cámara) ó 5 s de video (archivo)
                t_ahora = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                if t_b > 0 and t_ahora - t_b > 5.0:
                    # Timeout: si hubo alguna lectura, usar la mejor disponible
                    fallback = votador.consenso() or (votador._lecturas[-1] if votador.n else "")
                    if fallback:
                        placa_detectada = fallback
                        estado = ESTADO_REGISTRO
                        captura_guardada = False
                        print(f"[Placa] Timeout → mejor lectura: {placa_detectada}")
                    else:
                        print("[Sistema] Timeout sin placa. Reiniciando…")
                        estado = ESTADO_VELOCIDAD
                        cruzó_linea_a = cruzó_linea_b = False
                        t_a = t_b = 0.0
                    votador.reset()

            # ─── MÓDULO REGISTRO (JSON) + HUD FINAL ────────────────
            elif estado == ESTADO_REGISTRO:
                clasif = resultado_difuso["clasificacion"]
                horas  = resultado_difuso["horas_indisponibilidad"]

                # Guardar captura y registrar evento una sola vez
                if not captura_guardada:
                    ruta_cap = _guardar_captura(
                        frame, placa_detectada, velocidad_kmh, clasif, horas,
                    )
                    captura_guardada = True
                    print(f"[Captura] Guardada en: {ruta_cap}")

                    if not evento_registrado:
                        registrar_evento(placa_detectada, velocidad_kmh,
                                         clasif, horas, ruta_cap)
                        evento_registrado = True
                        print(f"[Registro] Evento guardado en registros/eventos.json")

                cv2.rectangle(frame_display, (10, 10), (500, 170), (0, 0, 0), -1)
                cv2.putText(frame_display, f"PLACA: {placa_detectada}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_display, f"VELOCIDAD: {velocidad_kmh} km/h",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                color_d = (0, 255, 0)
                if clasif == "multa":
                    color_d = (0, 0, 255)
                elif clasif == "normal":
                    color_d = (0, 255, 255)

                etiqueta = clasif.upper()
                if clasif == "multa":
                    etiqueta += f"  ({horas}h indisponible)"
                cv2.putText(frame_display, f"ESTADO: {etiqueta}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_d, 2)

                # Auto-reset 3 s tras registro → listo para siguiente auto
                t_ahora2 = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                if captura_guardada and t_ahora2 - t_b > 3.0:
                    estado = ESTADO_VELOCIDAD
                    cruzó_linea_a = cruzó_linea_b = False
                    t_a = t_b = 0.0
                    placa_detectada = ""
                    evento_registrado = False
                    captura_guardada  = False
                    resultado_difuso  = None
                    velocidad_kmh     = 0.0
                    worker.reset_lecturas()
                    print("[Sistema] Listo para siguiente vehículo…")

            # ─── HUD inferior ──────────────────────────────────────
            cv2.putText(frame_display, "F=fullscreen  R=reiniciar  ESC=salir",
                        (10, alto - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(WIN_NAME, frame_display)

            # Para archivos de video: respetar el FPS original (no reproducir a máxima velocidad)
            tecla = cv2.waitKey(delay_ms) & 0xFF
            if tecla == 27:           # ESC
                break
            elif tecla in (ord("f"), ord("F")):
                pantalla_completa = not pantalla_completa
                flag = cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, flag)
            elif tecla in (ord("r"), ord("R")):
                estado = ESTADO_VELOCIDAD
                cruzó_linea_a = cruzó_linea_b = False
                t_a = t_b = 0.0
                placa_detectada = ""
                evento_registrado = False
                resultado_difuso = None
                velocidad_kmh   = 0.0
                captura_guardada = False
                votador.reset()
                worker.reset_lecturas()
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
    evento = registrar_evento(
        placa_prueba, velocidad_prueba,
        resultado["clasificacion"], resultado["horas_indisponibilidad"],
    )
    print(f"✅ Evento registrado en registros/eventos.json: {evento}")


# ----------------------------------------------------------------
#  Entrada
# ----------------------------------------------------------------

def _uso():
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    args = sys.argv[1:]
    modo = args[0] if args else "demo"

    if modo in ("-h", "--help", "ayuda"):
        _uso()

    elif modo == "demo":
        demo_sin_camara("ABC-1234", 35.0)
        demo_sin_camara("XYZ-4567", 18.0)
        demo_sin_camara("KLM-1234", 25.0)

    elif modo == "video":
        if len(args) < 2:
            print("[ERROR] Especifica la ruta del video: python main.py video <ruta>")
            sys.exit(1)
        ruta = args[1]
        if not os.path.isfile(ruta):
            print(f"[ERROR] Archivo no encontrado: {ruta}")
            sys.exit(1)
        print(f"[Sistema] Fuente: archivo de video → {ruta}")
        procesar_vehiculo(cam_url=ruta)

    elif modo == "camara":
        indice = int(args[1]) if len(args) > 1 else 0
        print(f"[Sistema] Fuente: cámara física → /dev/video{indice} (índice {indice})")
        procesar_vehiculo(cam_url=indice)

    elif modo == "digital":
        fuente = args[1] if len(args) > 1 else URL_STREAM
        if isinstance(fuente, str) and fuente.isdigit():
            fuente = int(fuente)
        print(f"[Sistema] Fuente: cámara digital → {fuente}")
        procesar_vehiculo(cam_url=fuente)

    # ── alias legacy ──
    elif modo == "live":
        print("[Sistema] Fuente: DroidCam (live) → URL_STREAM")
        procesar_vehiculo(cam_url=URL_STREAM)

    elif modo == "laptop":
        print("[Sistema] Fuente: cámara laptop → índice 0")
        procesar_vehiculo(cam_url=0)

    else:
        print(f"[ERROR] Modo desconocido: '{modo}'")
        _uso()
