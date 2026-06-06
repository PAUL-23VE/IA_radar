import threading
import time
import cv2
import numpy as np

from utils.reconocedor import RecognitionWorker, VotadorPlaca
from utils.registro import registrar_evento
from velocidad.geometria import lado_linea, R_ENDPOINT
from velocidad.logica_difusa import clasificar_velocidad
from main import _abrir_camara, _guardar_captura, DISTANCIA_REFERENCIA_METROS, URL_STREAM
from api.email_service import enviar_notificacion_asincrona

ESTADO_VELOCIDAD = 0
ESTADO_PLACA     = 1
ESTADO_REGISTRO  = 2

class RadarPipeline:
    def __init__(self):
        self.running = False
        self.thread = None
        self.cap = None
        
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.estado_lineas = {
            "linea_a": {"x1": 64, "y1": 384, "x2": 448, "y2": 96},
            "linea_b": {"x1": 160, "y1": 432, "x2": 544, "y2": 144},
            "drag": None,
        }
        
        self.worker = RecognitionWorker()
        self.votador = VotadorPlaca(min_votos=3, conf_min=0.35)
        
        self.event_callbacks = [] # Callbacks func(type, payload)
        
        # State
        self.estado = ESTADO_VELOCIDAD
        self.velocidad_kmh = 0.0
        self.placa_detectada = ""
        self.resultado_difuso = None
        self.captura_guardada = False
        self.evento_registrado = False
        
        # Controles de reproducción
        self.paused = False
        self.playback_speed = 1.0
        self.fuente_actual = None
        self.es_archivo = False
        self.veh_bbox = None       # bbox del vehículo al cruzar línea B (para recortar OCR)
        self.ocr_offset = (0, 0)   # offset del recorte para dibujar bbox correctamente

    def on_event(self, cb):
        self.event_callbacks.append(cb)
        
    def _emit(self, event_type, payload):
        for cb in self.event_callbacks:
            cb(event_type, payload)

    def set_lines(self, linea_a, linea_b):
        self.estado_lineas["linea_a"] = linea_a
        self.estado_lineas["linea_b"] = linea_b

    def get_lines(self):
        return {
            "linea_a": self.estado_lineas["linea_a"],
            "linea_b": self.estado_lineas["linea_b"]
        }

    def start(self, fuente=URL_STREAM, distancia_m=DISTANCIA_REFERENCIA_METROS):
        if self.running:
            return False
            
        self.fuente_actual = fuente
        self.es_archivo = isinstance(fuente, str) and not fuente.startswith("http")
        self.paused = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.cap = _abrir_camara(fuente)
        if self.cap is None:
            return False
            
        ret, frame_init = self.cap.read()
        if not ret:
            self.cap.release()
            return False
            
        alto, ancho = frame_init.shape[:2]
        
        # Iniciar valores por defecto en base al ancho/alto real
        self.estado_lineas["linea_a"] = {
            "x1": int(ancho * 0.10), "y1": int(alto * 0.80),
            "x2": int(ancho * 0.70), "y2": int(alto * 0.20)
        }
        self.estado_lineas["linea_b"] = {
            "x1": int(ancho * 0.25), "y1": int(alto * 0.90),
            "x2": int(ancho * 0.85), "y2": int(alto * 0.30)
        }

        self.running = True
        self.worker.start()
        self.thread = threading.Thread(target=self._process_loop, args=(fuente, distancia_m, alto, ancho), daemon=True)
        self.thread.start()
        self._emit("status", {"state": "started", "source": str(fuente)})
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.worker.stop()
        if self.cap:
            self.cap.release()
        self.fuente_actual = None
        self.paused = False
        self._emit("status", {"state": "stopped"})

    def toggle_pause(self):
        self.paused = not getattr(self, 'paused', False)
        return self.paused

    def seek(self, delta_segundos):
        if not self.running or not self.cap or not getattr(self, 'es_archivo', False):
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        delta_frames = delta_segundos * fps
        nueva_pos = max(0.0, pos_frames + delta_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, nueva_pos)
        
        # Le decimos al loop principal que reinicie su estado
        self.needs_reset = True
        return True

    def restart_video(self):
        return self.seek(-9999999) # Va a cero

    def get_status(self):
        return {
            "running": getattr(self, 'running', False),
            "paused": getattr(self, 'paused', False),
            "fuente": getattr(self, 'fuente_actual', None),
            "es_archivo": getattr(self, 'es_archivo', False)
        }

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame_copy = self.current_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame_copy)
        return buffer.tobytes() if ret else None

    def _nuevo_track(self):
        return {
            "crossed_a": False, "crossed_b": False,
            "t_a": 0.0, "t_b": 0.0, "speed": 0.0,
            "prev_a": None, "prev_b": None,
            "votador": VotadorPlaca(min_votos=3, conf_min=0.35),
            "registrado": False, "placa": "",
            "last_seen": 0, "t_b_real": 0.0, "medido": False,
        }

    def _process_loop(self, fuente, distancia_m, alto, ancho):
        """
        Pipeline multi-vehiculo:
          - yolo11n (COCO) rastrea TODOS los vehiculos con ByteTrack (IDs estables)
          - cada track mide velocidad al cruzar linea A -> linea B
          - best.pt + CNN leen la placa de cada vehiculo cercano (voto por track)
          - al cruzar B con placa, se registra el evento + logica difusa
        Maneja varios autos a la vez; los parqueados nunca cruzan -> no disparan.
        """
        from ultralytics import YOLO
        from inferencia import reconocer_placa

        es_archivo = isinstance(fuente, str) and not fuente.startswith("http")
        fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(1, int(1000 / fps_video)) if es_archivo else 1

        # YOLO de vehiculos (COCO). Se cachea en la instancia para no recargar.
        if getattr(self, "_veh_yolo", None) is None:
            self._veh_yolo = YOLO("yolo11n.pt")
        veh_yolo = self._veh_yolo
        VEH_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)
        MIN_H_OCR = 0.10 * alto       # solo OCR de vehiculos suficientemente cercanos

        tracks = {}
        frame_idx = 0

        self.velocidad_kmh = 0.0
        self.placa_detectada = ""
        self.resultado_difuso = None
        self.needs_reset = False

        while self.running:
            if getattr(self, 'needs_reset', False):
                self.needs_reset = False
                tracks.clear()
                veh_yolo.predictor = None  # reinicia el estado del tracker ByteTrack
                self.velocidad_kmh = 0.0
                self.placa_detectada = ""
                self.resultado_difuso = None
                self._emit("status", {"state": "ready"})

            if getattr(self, 'paused', False):
                time.sleep(0.1)
                continue

            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self._emit("status", {"state": "ended"})
                break

            frame_idx += 1
            ahora = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()

            la = self.estado_lineas["linea_a"]
            lb = self.estado_lineas["linea_b"]
            frame_display = frame.copy()

            # ── Tracking de vehiculos ────────────────────────────────────────
            res = veh_yolo.track(frame, classes=VEH_CLASSES, conf=0.35,
                                 persist=True, verbose=False, tracker="bytetrack.yaml")[0]

            vivos = set()
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.id.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(tid)
                    vivos.add(tid)
                    tr = tracks.get(tid) or tracks.setdefault(tid, self._nuevo_track())
                    tr["last_seen"] = frame_idx

                    # Punto de contacto con el suelo: centro-inferior de la caja.
                    cx = (x1 + x2) // 2
                    cy = y2
                    bh = y2 - y1

                    # OCR de placa si el vehiculo esta cerca y aun no registrado.
                    if not tr["registrado"] and bh >= MIN_H_OCR:
                        veh_crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if veh_crop.size:
                            placa, _bb, conf = reconocer_placa(veh_crop)
                            tr["votador"].agregar(placa, conf)

                    # Cruce de cada linea de forma INDEPENDIENTE (cualquier orden
                    # ni direccion). El auto puede entrar ya pasado A, o ir al reves.
                    la_s = lado_linea(cx, cy, la["x1"], la["y1"], la["x2"], la["y2"])
                    lb_s = lado_linea(cx, cy, lb["x1"], lb["y1"], lb["x2"], lb["y2"])
                    if not tr["crossed_a"]:
                        if tr["prev_a"] is not None and tr["prev_a"] * la_s < 0:
                            tr["t_a"] = ahora
                            tr["crossed_a"] = True
                        tr["prev_a"] = la_s
                    if not tr["crossed_b"]:
                        if tr["prev_b"] is not None and tr["prev_b"] * lb_s < 0:
                            tr["t_b"] = ahora
                            tr["crossed_b"] = True
                        tr["prev_b"] = lb_s

                    # Al cruzar AMBAS lineas (una sola vez) -> calcula velocidad.
                    if tr["crossed_a"] and tr["crossed_b"] and not tr["medido"]:
                        tr["medido"] = True
                        dt = abs(tr["t_b"] - tr["t_a"])
                        tr["speed"] = round((distancia_m / dt) * 3.6, 2) if dt > 0 else 0.0
                        tr["t_b_real"] = time.time()
                        if tr["speed"] > 0:
                            self._emit("event", {"type": "velocidad", "velocidad": tr["speed"]})

                    # Registro: medido con velocidad valida y hay placa (o grace 1.5s).
                    if tr["medido"] and tr["speed"] > 0 and not tr["registrado"]:
                        placa_final = tr["votador"].consenso() or tr["votador"].mejor_lectura()[0]
                        grace = tr["t_b_real"] > 0 and time.time() - tr["t_b_real"] > 1.5
                        if placa_final or grace:
                            self._registrar_track(frame, tr, placa_final)

                    # Dibujo: color segun estado del track.
                    if tr["registrado"]:
                        color = (0, 200, 0)
                        etiqueta = f"{tr['placa']} {tr['speed']:.0f}km/h"
                    elif tr["crossed_a"] or tr["crossed_b"]:
                        color = (0, 200, 255)
                        cons = tr["votador"].consenso()
                        etiqueta = cons or "midiendo..."
                    else:
                        color = (180, 180, 180)
                        etiqueta = ""
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                    if etiqueta:
                        cv2.putText(frame_display, etiqueta, (x1, max(18, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Limpieza de tracks perdidos (no vistos en 30 frames).
            for tid in [t for t, d in tracks.items()
                        if frame_idx - d["last_seen"] > 30]:
                del tracks[tid]

            with self.frame_lock:
                self.current_frame = frame_display

            if es_archivo:
                elapsed = time.time() - start_time
                target_delay = delay_ms / (1000.0 * getattr(self, 'playback_speed', 1.0))
                sleep_time = target_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _registrar_track(self, frame, tr, placa_final):
        """Guarda captura, registra evento y notifica para un track que cruzo B."""
        tr["registrado"] = True
        tr["placa"] = placa_final or "SIN-LECTURA"
        difuso = clasificar_velocidad(tr["speed"])
        clasif = difuso["clasificacion"]
        horas = difuso["horas_indisponibilidad"]

        # Estado para el panel lateral (ultimo evento).
        self.velocidad_kmh = tr["speed"]
        self.placa_detectada = tr["placa"]
        self.resultado_difuso = difuso

        ruta_cap = _guardar_captura(frame, tr["placa"], tr["speed"], clasif, horas)
        registrar_evento(tr["placa"], tr["speed"], clasif, horas, ruta_cap)
        datos_correo = {
            "placa": tr["placa"], "velocidad_kmh": tr["speed"],
            "clasificacion": clasif, "horas": horas, "ruta_captura": ruta_cap,
        }
        threading.Thread(target=enviar_notificacion_asincrona,
                         args=(datos_correo,), daemon=True).start()
        self._emit("event", {"type": "placa", "placa": tr["placa"]})
        self._emit("event", {"type": "registro_guardado", "placa": tr["placa"],
                             "velocidad": tr["speed"], "clasificacion": clasif,
                             "horas": horas, "captura": ruta_cap})

# Instancia global para que el servidor la use
pipeline = RadarPipeline()
