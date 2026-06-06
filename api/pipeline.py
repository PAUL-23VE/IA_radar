import threading
import time
import cv2
import numpy as np

from utils.reconocedor import RecognitionWorker, VotadorPlaca
from utils.registro import registrar_evento
from velocidad.geometria import lado_linea, R_ENDPOINT
from velocidad.logica_difusa import clasificar_velocidad
from main import _abrir_camara, _guardar_captura, DISTANCIA_REFERENCIA_METROS, URL_STREAM

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
        self.votador = VotadorPlaca()
        
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
        pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        nueva_pos = max(0.0, pos + delta_segundos * 1000.0)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, nueva_pos)
        
        # Resetear estado temporal para no mezclar mediciones
        self.estado = ESTADO_VELOCIDAD
        self.velocidad_kmh = 0.0
        self.placa_detectada = ""
        self.resultado_difuso = None
        self.worker.reset_lecturas()
        self.votador.reset()
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

    def _process_loop(self, fuente, distancia_m, alto, ancho):
        es_archivo = isinstance(fuente, str) and not fuente.startswith("http")
        fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(1, int(1000 / fps_video)) if es_archivo else 1
        
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        
        t_a = t_b = 0.0
        cruzó_linea_a = cruzó_linea_b = False
        prev_lado_a = prev_lado_b = None
        last_veh_bbox = None
        
        self.estado = ESTADO_VELOCIDAD
        self.velocidad_kmh = 0.0
        self.placa_detectada = ""
        self.resultado_difuso = None
        self.captura_guardada = False
        self.evento_registrado = False
        
        while self.running:
            if getattr(self, 'paused', False):
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # Video ended or camera disconnected
                self.running = False
                self._emit("status", {"state": "ended"})
                break

            la = self.estado_lineas["linea_a"]
            lb = self.estado_lineas["linea_b"]

            frame_display = frame.copy()

            # Se dibujan las lineas de forma interactiva en el frontend.
            # Comentado para evitar lineas dobles:
            # for nombre, ln, color_base in [("a", la, (255, 100, 0)), ("b", lb, (0, 0, 255))]:
            #     color = color_base
            #     p1 = (ln["x1"], ln["y1"])
            #     p2 = (ln["x2"], ln["y2"])
            #     cv2.line(frame_display, p1, p2, color, 3)
            #     cv2.circle(frame_display, p1, R_ENDPOINT, color, -1)
            #     cv2.circle(frame_display, p2, R_ENDPOINT, color, -1)

            if self.estado == ESTADO_VELOCIDAD:
                mascara = fgbg.apply(frame)
                _, mask_bin = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                veh_cx = veh_cy = None
                mejor_area = 0
                for cnt in contornos:
                    area = cv2.contourArea(cnt)
                    if area > 2000 and area > mejor_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        if bh < 25 or bw > bh * 8:
                            continue
                        veh_cx = bx + bw // 2
                        veh_cy = by + bh // 2
                        mejor_area = area
                        last_veh_bbox = (bx, by, bw, bh)
                        cv2.rectangle(frame_display, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

                if veh_cx is not None and veh_cy is not None:
                    lado_a = lado_linea(veh_cx, veh_cy, la["x1"], la["y1"], la["x2"], la["y2"])
                    lado_b = lado_linea(veh_cx, veh_cy, lb["x1"], lb["y1"], lb["x2"], lb["y2"])

                    if not cruzó_linea_a:
                        if prev_lado_a is not None and prev_lado_a < 0 and lado_a >= 0:
                            t_a = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_a = True
                        prev_lado_a = lado_a
                        
                    if cruzó_linea_a and not cruzó_linea_b:
                        if prev_lado_b is not None and prev_lado_b < 0 and lado_b >= 0:
                            t_b = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_b = True
                            dt = t_b - t_a
                            if dt > 0:
                                self.velocidad_kmh = round((distancia_m / dt) * 3.6, 2)
                            self.resultado_difuso = clasificar_velocidad(self.velocidad_kmh)
                            self.veh_bbox = last_veh_bbox
                            self.ocr_offset = (0, 0)
                            self.estado = ESTADO_PLACA
                            self._emit("event", {"type": "velocidad", "velocidad": self.velocidad_kmh, "clasificacion": self.resultado_difuso})
                        prev_lado_b = lado_b

            elif self.estado == ESTADO_PLACA:
                cv2.rectangle(frame_display, (10, 10), (450, 50), (0,0,0), -1)
                cv2.putText(frame_display, f"V: {self.velocidad_kmh} km/h - BUSCANDO PLACA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Recortar al área del vehículo para que YOLO no detecte placas
                # de autos estáticos en el fondo del frame completo.
                frame_to_ocr = frame
                if self.veh_bbox is not None:
                    bx, by, bw, bh = self.veh_bbox
                    pad_x = int(bw * 0.7)
                    pad_y = int(bh * 0.7)
                    cx1 = max(0, bx - pad_x)
                    cy1 = max(0, by - pad_y)
                    cx2 = min(ancho, bx + bw + pad_x)
                    cy2 = min(alto, by + bh + pad_y)
                    if cx2 > cx1 and cy2 > cy1:
                        frame_to_ocr = frame[cy1:cy2, cx1:cx2]
                        self.ocr_offset = (cx1, cy1)
                self.worker.submit(frame_to_ocr)

                placa_aceptada = ""
                for p, bb, c in self.worker.drenar_lecturas():
                    # Fast path: una lectura de alta confianza es suficiente
                    if p and c >= 0.70:
                        placa_aceptada = p
                        break
                    self.votador.agregar(p, c)

                _, bbox, _ = self.worker.get_result()
                if bbox:
                    ox, oy = self.ocr_offset
                    x, y, w, h = bbox
                    cv2.rectangle(frame_display, (x + ox, y + oy), (x + ox + w, y + oy + h), (0, 255, 0), 2)

                if not placa_aceptada:
                    placa_aceptada = self.votador.consenso()

                if placa_aceptada:
                    self.placa_detectada = placa_aceptada
                    self.estado = ESTADO_REGISTRO
                    self.captura_guardada = False
                    self.votador.reset()
                    self._emit("event", {"type": "placa", "placa": self.placa_detectada})

                t_ahora = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                if t_b > 0 and t_ahora - t_b > 2.5:
                    mejor, _ = self.votador.mejor_lectura()
                    fallback = self.votador.consenso() or mejor
                    if fallback:
                        self.placa_detectada = fallback
                        self.estado = ESTADO_REGISTRO
                        self.captura_guardada = False
                        self._emit("event", {"type": "placa", "placa": self.placa_detectada})
                    else:
                        self.estado = ESTADO_VELOCIDAD
                        cruzó_linea_a = cruzó_linea_b = False
                        t_a = t_b = 0.0
                        last_veh_bbox = None
                        self.veh_bbox = None
                        self.ocr_offset = (0, 0)
                        self._emit("status", {"state": "timeout_placa"})
                    self.votador.reset()

            elif self.estado == ESTADO_REGISTRO:
                clasif = self.resultado_difuso["clasificacion"]
                horas = self.resultado_difuso["horas_indisponibilidad"]

                if not self.captura_guardada:
                    ruta_cap = _guardar_captura(frame, self.placa_detectada, self.velocidad_kmh, clasif, horas)
                    self.captura_guardada = True
                    if not self.evento_registrado:
                        registrar_evento(self.placa_detectada, self.velocidad_kmh, clasif, horas, ruta_cap)
                        self.evento_registrado = True
                        self._emit("event", {"type": "registro_guardado", "placa": self.placa_detectada, "velocidad": self.velocidad_kmh, "clasificacion": clasif, "horas": horas, "captura": ruta_cap})

                cv2.rectangle(frame_display, (10, 10), (500, 170), (0, 0, 0), -1)
                cv2.putText(frame_display, f"PLACA: {self.placa_detectada}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_display, f"VELOCIDAD: {self.velocidad_kmh} km/h", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"ESTADO: {clasif.upper()}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                t_ahora2 = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                if self.captura_guardada and t_ahora2 - t_b > 3.0:
                    self.estado = ESTADO_VELOCIDAD
                    cruzó_linea_a = cruzó_linea_b = False
                    t_a = t_b = 0.0
                    prev_lado_a = prev_lado_b = None
                    last_veh_bbox = None
                    self.veh_bbox = None
                    self.ocr_offset = (0, 0)
                    self.placa_detectada = ""
                    self.evento_registrado = False
                    self.captura_guardada = False
                    self.resultado_difuso = None
                    self.velocidad_kmh = 0.0
                    self.worker.reset_lecturas()
                    self._emit("status", {"state": "ready"})

            with self.frame_lock:
                self.current_frame = frame_display

            if es_archivo:
                elapsed = time.time() - start_time
                sleep_time = (delay_ms / 1000.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

# Instancia global para que el servidor la use
pipeline = RadarPipeline()
