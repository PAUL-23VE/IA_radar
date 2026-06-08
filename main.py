"""
main.py
Pipeline completo del sistema de reconocimiento de placas.

Flujo:
  1. Captura frame desde la fuente seleccionada — hilo principal
  2. Hilo de reconocimiento (RecognitionWorker): YOLO + CNN de caracteres
  3. Mide velocidad con dos líneas virtuales + lógica difusa (sanción en horas)
  4. Registra el evento (placa, velocidad, multa, captura) en outputs/registros/eventos.json

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

import sys
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "velocidad"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from inferencia    import reconocer_placa, validar_formato_placa
from logica_difusa import clasificar_velocidad
from registro      import registrar_evento
from camara        import URL_STREAM
from reconocedor   import RecognitionWorker, VotadorPlaca
from geometria     import R_ENDPOINT, lado_linea, callback_mouse_lineas
from config        import settings

DISTANCIA_REFERENCIA_METROS = settings.DISTANCIA_REFERENCIA_METROS

ESTADO_VELOCIDAD = 0
ESTADO_PLACA     = 1
ESTADO_REGISTRO  = 2

WIN_NAME     = "Sistema Integrado de Placas"
DIR_CAPTURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "capturas")





# ----------------------------------------------------------------
#  Guardar captura al detectar placa
# ----------------------------------------------------------------

def _guardar_captura(frame: np.ndarray, placa: str, velocidad: float,
                     clasificacion: str, horas: int) -> str:
    from velocidad.logica_difusa import formatear_tiempo_sancion
    os.makedirs(DIR_CAPTURAS, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre    = f"{placa.replace('-', '')}_{ts}.jpg"
    ruta      = os.path.join(DIR_CAPTURAS, nombre)

    img = frame.copy()
    h, w = img.shape[:2]

    # Fondo semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    color_estado = {
        "multa":       (0, 0, 255),
        "advertencia": (0, 165, 255),
        "normal":      (0, 255, 255),
        "felicitacion":(0, 255, 0),
    }.get(clasificacion, (200, 200, 200))

    cv2.putText(img, f"PLACA: {placa}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, f"VELOCIDAD: {velocidad:.1f} km/h",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    etiqueta = clasificacion.upper()
    if horas > 0:
        etiqueta += f"  [{formatear_tiempo_sancion(horas)}]"
    cv2.putText(img, f"ESTADO: {etiqueta}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_estado, 2)

    cv2.putText(img, ts, (w - 185, h - 10),
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

    # Dos diagonales paralelas: vector idéntico (0.60W, -0.60H) → paralelas garantizadas.
    # Van de esquina inferior-izquierda hacia superior-derecha, ~60% del ancho.
    estado_lineas = {
        "linea_a": {"x1": int(ancho * 0.10), "y1": int(alto * 0.80),
                    "x2": int(ancho * 0.70), "y2": int(alto * 0.20)},
        "linea_b": {"x1": int(ancho * 0.25), "y1": int(alto * 0.90),
                    "x2": int(ancho * 0.85), "y2": int(alto * 0.30)},
        "drag": None,
    }

    # Ventana redimensionable; arranca en pantalla completa
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN_NAME, callback_mouse_lineas, estado_lineas)

    pantalla_completa = True

    estado           = ESTADO_VELOCIDAD
    model_vehiculos  = YOLO('yolo11n.pt')
    t_a = t_b        = 0.0
    cruzó_linea_a    = False
    cruzó_linea_b    = False
    prev_lado_a      = None   # lado previo respecto a línea A (para detectar cruce real)
    prev_lado_b      = None   # lado previo respecto a línea B
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

            # ─── Líneas siempre visibles (todos los estados) ───────
            drag_nombre = estado_lineas["drag"][0] if estado_lineas["drag"] else None
            for nombre, ln, color_base in [("a", la, (255, 100, 0)), ("b", lb, (0, 0, 255))]:
                color = (0, 180, 255) if drag_nombre == nombre else color_base
                p1 = (ln["x1"], ln["y1"]); p2 = (ln["x2"], ln["y2"])
                cv2.line(frame_display, p1, p2, color, 3)
                cv2.circle(frame_display, p1, R_ENDPOINT, color, -1)
                cv2.circle(frame_display, p2, R_ENDPOINT, color, -1)
                cv2.putText(frame_display, f"Linea {nombre.upper()}",
                            (ln["x1"] + 8, ln["y1"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

            # ─── MÓDULO VELOCIDAD ──────────────────────────────────
            if estado == ESTADO_VELOCIDAD:
                # 2=car, 3=motorcycle, 5=bus, 7=truck
                resultados = model_vehiculos.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
                
                veh_cx = veh_cy = None
                mejor_area = 0
                
                if resultados[0].boxes and resultados[0].boxes.id is not None:
                    cajas = resultados[0].boxes.xyxy.cpu().numpy()
                    ids = resultados[0].boxes.id.cpu().numpy()
                    
                    for caja, track_id in zip(cajas, ids):
                        x1, y1, x2, y2 = caja
                        bw = x2 - x1
                        bh = y2 - y1
                        area = bw * bh
                        if area > mejor_area:
                            mejor_area = area
                            veh_cx = int(x1 + bw / 2)
                            veh_cy = int(y1 + bh / 2)
                            bx, by = int(x1), int(y1)
                            cv2.rectangle(frame_display, (bx, by), (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(frame_display, f"ID: {int(track_id)}", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.putText(frame_display, "MIDIENDO VELOCIDAD…", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if veh_cx is not None and veh_cy is not None:
                    lado_a = lado_linea(veh_cx, veh_cy,
                                         la["x1"], la["y1"], la["x2"], la["y2"])
                    lado_b = lado_linea(veh_cx, veh_cy,
                                         lb["x1"], lb["y1"], lb["x2"], lb["y2"])

                    # Cruce real: exige cambio de signo negativo→positivo
                    if not cruzó_linea_a:
                        if prev_lado_a is not None and prev_lado_a < 0 and lado_a >= 0:
                            t_a = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_a = True
                            print("[Velocidad] Cruzó Línea A")
                        prev_lado_a = lado_a
                    if cruzó_linea_a and not cruzó_linea_b:
                        if prev_lado_b is not None and prev_lado_b < 0 and lado_b >= 0:
                            t_b = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_b = True
                            dt = t_b - t_a
                            if dt > 0:
                                velocidad_kmh = round((distancia_m / dt) * 3.6, 2)
                            resultado_difuso = clasificar_velocidad(velocidad_kmh)
                            estado = ESTADO_PLACA
                            print(f"[Velocidad] {velocidad_kmh} km/h → {resultado_difuso['clasificacion']}")
                        prev_lado_b = lado_b

            # ─── MÓDULO PLACA (hilo worker) ────────────────────────
            elif estado == ESTADO_PLACA:
                cv2.putText(frame_display,
                            f"V: {velocidad_kmh} km/h  [{resultado_difuso['clasificacion'].upper()}]",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame_display, "BUSCANDO PLACA…",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Enviar los frames al worker directamente (YOLO tracking asume que el auto existe)
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
                placa_consenso_cruda = votador.consenso()
                placa_valida = validar_formato_placa(placa_consenso_cruda)
                if placa_valida:
                    placa_detectada = placa_valida
                    estado = ESTADO_REGISTRO
                    captura_guardada = False
                    print(f"[Placa] Consenso ({votador.n} lecturas): {placa_detectada}")
                    votador.reset()

                # Timeout: 5 s reales (cámara) ó 5 s de video (archivo)
                t_ahora = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                if t_b > 0 and t_ahora - t_b > 5.0:
                    # Timeout: si hubo alguna lectura, usar la mejor disponible
                    fallback_cruda = votador.consenso() or (votador._lecturas[-1] if votador.n else "")
                    placa_valida = validar_formato_placa(fallback_cruda)
                    if placa_valida:
                        placa_detectada = placa_valida
                        estado = ESTADO_REGISTRO
                        captura_guardada = False
                        print(f"[Placa] Timeout → mejor lectura: {placa_detectada}")
                    else:
                        print(f"[Sistema] Timeout sin placa válida (crudo: {fallback_cruda}). Reiniciando…")
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
                        print(f"[Registro] Evento guardado en outputs/registros/eventos.json")

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
                    prev_lado_a = prev_lado_b = None
                    placa_detectada = ""
                    evento_registrado = False
                    captura_guardada  = False
                    resultado_difuso  = None
                    velocidad_kmh     = 0.0
                    worker.reset_lecturas()
                    print("[Sistema] Listo para siguiente vehículo…")

            # ─── HUD inferior ──────────────────────────────────────
            hint = "F=fullscreen  R=reiniciar  ESC=salir"
            if es_archivo:
                hint += "  ←/J=retroceder  →/L=avanzar"
            cv2.putText(frame_display, hint,
                        (10, alto - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(WIN_NAME, frame_display)

            tecla_raw = cv2.waitKey(delay_ms)
            tecla = tecla_raw & 0xFF if tecla_raw != -1 else 0xFF
            if tecla == 27:           # ESC
                break
            elif tecla in (ord("f"), ord("F")):
                pantalla_completa = not pantalla_completa
                flag = cv2.WINDOW_FULLSCREEN if pantalla_completa else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, flag)
            elif es_archivo and (tecla_raw in (81, 65361) or tecla == ord("j")):
                # Retroceder 5 s en el video
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, pos - 5000.0))
            elif es_archivo and (tecla_raw in (83, 65363) or tecla == ord("l")):
                # Avanzar 5 s en el video
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                cap.set(cv2.CAP_PROP_POS_MSEC, pos + 5000.0)
            elif tecla in (ord("r"), ord("R")):
                estado = ESTADO_VELOCIDAD
                cruzó_linea_a = cruzó_linea_b = False
                t_a = t_b = 0.0
                prev_lado_a = prev_lado_b = None
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
    print(f"✅ Evento registrado en outputs/registros/eventos.json: {evento}")


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

    elif modo == "web":
        print("\n" + "=" * 60)
        print("  INICIANDO SERVIDOR WEB (FastAPI)")
        print("=" * 60)
        import uvicorn
        uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)

    else:
        print(f"[ERROR] Modo desconocido: '{modo}'")
        _uso()
