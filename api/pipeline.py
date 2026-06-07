import threading
import time
import cv2
import numpy as np

from utils.reconocedor import RecognitionWorker, VotadorPlaca
from utils.registro import registrar_evento
from velocidad.geometria import lado_linea, R_ENDPOINT
from velocidad.logica_difusa import clasificar_velocidad, _render_inferencia_png
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
        
        # Lineas por defecto: dos lineas HORIZONTALES juntas en la zona cercana
        # (mitad-inferior del cuadro), donde los autos se ven grandes y la placa
        # es legible. Juntas = el auto cruza ambas en pocos frames -> el track
        # sobrevive (ByteTrack no pierde el ID) y se mide la velocidad aunque el
        # trafico vaya lento. El usuario las puede arrastrar para afinar el carril.
        self.estado_lineas["linea_a"] = {
            "x1": int(ancho * 0.12), "y1": int(alto * 0.55),
            "x2": int(ancho * 0.88), "y2": int(alto * 0.55)
        }
        self.estado_lineas["linea_b"] = {
            "x1": int(ancho * 0.10), "y1": int(alto * 0.72),
            "x2": int(ancho * 0.90), "y2": int(alto * 0.72)
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
            try:
                import torch
                if torch.cuda.is_available():
                    self._veh_yolo.to("cuda")   # CRITICO: si no, corre en CPU = lag
            except Exception:
                pass
        veh_yolo = self._veh_yolo
        VEH_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO)
        MIN_H_OCR = 0.10 * alto       # solo OCR de vehiculos suficientemente cercanos
        MIN_H_TRACK = 0.07 * alto     # ignora autos diminutos (lejanos) -> menos trabajo

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

            # ── Tracking de vehiculos (sobre frame REDUCIDo para ir 2-4x mas rapido)
            DET_SCALE = 0.5
            small = cv2.resize(frame, (0, 0), fx=DET_SCALE, fy=DET_SCALE)
            res = veh_yolo.track(small, classes=VEH_CLASSES, conf=0.35,
                                 persist=True, verbose=False, tracker="bytetrack.yaml")[0]
            inv = 1.0 / DET_SCALE

            vivos = set()
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.id.cpu().numpy()):
                    # Escalar caja de vuelta a coordenadas del frame original.
                    x1, y1, x2, y2 = (int(box[0]*inv), int(box[1]*inv),
                                      int(box[2]*inv), int(box[3]*inv))
                    tid = int(tid)
                    vivos.add(tid)
                    tr = tracks.get(tid) or tracks.setdefault(tid, self._nuevo_track())
                    tr["last_seen"] = frame_idx

                    # Punto de contacto con el suelo: centro-inferior de la caja.
                    cx = (x1 + x2) // 2
                    cy = y2
                    bh = y2 - y1

                    # ── Cruce de lineas — SIEMPRE, aun para autos lejanos/chicos ──
                    # Es barato (solo geometria) y DEBE seguirse desde lejos: los
                    # autos cruzan la linea A cuando aun estan pequenos (al fondo).
                    # Si se hiciera tras el gate de tamano, prev_a nunca se setearia
                    # mientras el auto es chico y se perderia el cruce de A -> el track
                    # solo cruzaria B -> nunca se mide -> sin evento/correo.
                    # Orden independiente: el auto puede cruzar A->B o B->A.
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

                    # Autos diminutos (lejanos): ya seguimos su cruce arriba; aqui
                    # solo evitamos el OCR/registro pesado (placa ilegible) y el
                    # dibujo grande. Su registro, si midio, lo hace el finalizador.
                    if bh < MIN_H_TRACK and not tr["registrado"]:
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (130, 130, 130), 1)
                        continue

                    # OCR de placa SOLO para autos en la zona de medicion (ya
                    # cruzaron una linea) y no registrados. Asi se SALTAN todos los
                    # autos PARQUEADOS (nunca cruzan) -> mucho menos lag.
                    # Crop del frame ORIGINAL full-res; alterna frames por track.
                    en_zona = tr["crossed_a"] or tr["crossed_b"]
                    if (en_zona and not tr["registrado"] and bh >= MIN_H_OCR
                            and (frame_idx + tid) % 2 == 0):
                        veh_crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if veh_crop.size:
                            placa, pbb, conf = reconocer_placa(veh_crop)
                            # Pondera por TAMAÑO de placa: frames de cerca (placa
                            # grande, nitida) pesan mas que los lejanos (chica, borrosa).
                            ph = pbb[3] if pbb else 0
                            size_w = max(0.3, min(1.6, ph / 40.0))
                            tr["votador"].agregar(placa, conf * size_w)

                    # Registro rapido: cruzo ambas, hay consenso rico (>=4 votos de
                    # cerca) -> registra mientras el auto sigue visible.
                    if (tr["medido"] and tr["speed"] > 0 and not tr["registrado"]
                            and tr["votador"].consenso() and tr["votador"].n >= 4):
                        self._registrar_track(frame, tr, tr["votador"].consenso())

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

            # Finalizador: tracks medidos que NO se registraron rapido (auto salio
            # del frame o junto pocos votos). Registra con lo mejor disponible al
            # perderse (no visto este frame) o tras grace de 1.2s.
            for d in tracks.values():
                if d["medido"] and d["speed"] > 0 and not d["registrado"]:
                    perdido = d["last_seen"] != frame_idx
                    grace = d["t_b_real"] > 0 and time.time() - d["t_b_real"] > 1.2
                    if perdido or grace:
                        placa_final = d["votador"].consenso() or d["votador"].mejor_lectura()[0]
                        if placa_final:
                            self._registrar_track(frame, d, placa_final)

            # Limpieza de tracks perdidos (no vistos en 30 frames).
            for tid in [t for t, d in tracks.items()
                        if frame_idx - d["last_seen"] > 30]:
                del tracks[tid]

            with self.frame_lock:
                self.current_frame = frame_display

            if es_archivo:
                elapsed = time.time() - start_time
                target_delay = delay_ms / (1000.0 * getattr(self, 'playback_speed', 1.0))
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                else:
                    # Procesar es mas lento que el ritmo real -> saltar frames para
                    # NO atrasarse (el video sigue a velocidad correcta). La velocidad
                    # medida usa POS_MSEC, asi que sigue siendo exacta pese al salto.
                    n_skip = min(10, int(elapsed / target_delay) - 1)
                    for _ in range(max(0, n_skip)):
                        if not self.cap.grab():
                            break

    def _registrar_track(self, frame, tr, placa_final):
        """Guarda captura, registra evento y notifica para un track que cruzo B."""
        tr["registrado"] = True
        tr["placa"] = placa_final or "SIN-LECTURA"
        difuso = clasificar_velocidad(tr["speed"])
        clasif = difuso["clasificacion"]
        horas = difuso["horas_indisponibilidad"]
        tiempo_sancion = difuso["tiempo_sancion"]

        # Estado para el panel lateral (ultimo evento).
        self.velocidad_kmh = tr["speed"]
        self.placa_detectada = tr["placa"]
        self.resultado_difuso = difuso

        ruta_cap = _guardar_captura(frame, tr["placa"], tr["speed"], clasif, horas)
        registrar_evento(tr["placa"], tr["speed"], clasif, horas, ruta_cap)

        # Gráfica del razonamiento difuso (Mamdani) — se renderiza UNA vez:
        #   - bytes → archivo PNG para adjuntar inline en el correo (cid)
        #   - bytes → data-URI base64 para el modal del frontend
        chart_uri = ""
        ruta_grafica = ""
        try:
            png = _render_inferencia_png(tr["speed"])
            import base64 as _b64
            chart_uri = "data:image/png;base64," + _b64.b64encode(png).decode("ascii")
            if ruta_cap:
                import os
                ruta_grafica = os.path.splitext(ruta_cap)[0] + "_difuso.png"
                with open(ruta_grafica, "wb") as _f:
                    _f.write(png)
        except Exception as e:
            print(f"[difuso] no se pudo graficar: {e}")

        datos_correo = {
            "placa": tr["placa"], "velocidad_kmh": tr["speed"],
            "clasificacion": clasif, "horas": horas,
            "tiempo_sancion": tiempo_sancion, "ruta_captura": ruta_cap,
            "ruta_grafica": ruta_grafica,
        }
        threading.Thread(target=enviar_notificacion_asincrona,
                         args=(datos_correo,), daemon=True).start()

        # Miniatura JPEG base64 del frame (resolución media para la grilla del frontend).
        thumb = ""
        try:
            h, w = frame.shape[:2]
            tw = 320
            th = max(1, int(h * tw / w))
            small = cv2.resize(frame, (tw, th))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                import base64
                thumb = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
        except Exception:
            pass

        self._emit("event", {"type": "placa", "placa": tr["placa"]})
        self._emit("event", {
            "type": "registro_guardado",
            "placa": tr["placa"],
            "velocidad": tr["speed"],
            "clasificacion": clasif,
            "horas": horas,
            "tiempo_sancion": tiempo_sancion,
            "captura": ruta_cap,
            "thumb": thumb,
            "chart": chart_uri,
            "ts": time.time(),
        })

# Instancia global para que el servidor la use
pipeline = RadarPipeline()
