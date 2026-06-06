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

    def _process_loop(self, fuente, distancia_m, alto, ancho):
        es_archivo = isinstance(fuente, str) and not fuente.startswith("http")
        fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(1, int(1000 / fps_video)) if es_archivo else 1
        
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        
        t_a = t_b = 0.0
        t_placa_inicio = 0.0
        t_registro_inicio = 0.0
        cruzó_linea_a = cruzó_linea_b = False
        prev_lado_a = prev_lado_b = None
        last_veh_bbox = None
        
        self.estado = ESTADO_VELOCIDAD
        self.velocidad_kmh = 0.0
        self.placa_detectada = ""
        self.resultado_difuso = None
        self.captura_guardada = False
        self.evento_registrado = False
        
        self.needs_reset = False
        
        while self.running:
            if getattr(self, 'needs_reset', False):
                self.needs_reset = False
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
                t_a = t_b = 0.0
                t_placa_inicio = 0.0
                t_registro_inicio = 0.0
                cruzó_linea_a = cruzó_linea_b = False
                prev_lado_a = prev_lado_b = None
                last_veh_bbox = None
                self.estado = ESTADO_VELOCIDAD
                self.velocidad_kmh = 0.0
                self.placa_detectada = ""
                self.resultado_difuso = None
                self.captura_guardada = False
                self.evento_registrado = False
                self.worker.reset_lecturas()
                if hasattr(self, 'votador'):
                    self.votador.reset()
                self._emit("status", {"state": "ready"})
                
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
                # Downscale for faster MOG2
                scale = 0.5
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                mascara = fgbg.apply(small_frame)
                _, mask_bin = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                veh_cx = veh_cy = None
                mejor_area = 0
                for cnt in contornos:
                    area = cv2.contourArea(cnt)
                    # Adjust area threshold since the image is scaled by 0.5 (area is 0.25)
                    if area > 500 and area > mejor_area:
                        bx_s, by_s, bw_s, bh_s = cv2.boundingRect(cnt)
                        # Scale bounding box back to original coordinates
                        bx, by = int(bx_s / scale), int(by_s / scale)
                        bw, bh = int(bw_s / scale), int(bh_s / scale)
                        
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
                        # Primera detección o cruce en cualquier dirección
                        if prev_lado_a is None or (prev_lado_a * lado_a < 0):
                            t_a = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_a = True
                        prev_lado_a = lado_a

                    if cruzó_linea_a and not cruzó_linea_b:
                        if prev_lado_b is not None and prev_lado_b * lado_b < 0:
                            t_b = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if es_archivo else time.time()
                            cruzó_linea_b = True
                            dt = abs(t_b - t_a)
                            if dt > 0:
                                self.velocidad_kmh = round((distancia_m / dt) * 3.6, 2)
                            self.resultado_difuso = clasificar_velocidad(self.velocidad_kmh)
                            self.veh_bbox = last_veh_bbox
                            self.ocr_offset = (0, 0)
                            self.estado = ESTADO_PLACA
                            t_placa_inicio = time.time()  # reloj real para el timeout
                            self._emit("event", {"type": "velocidad", "velocidad": self.velocidad_kmh, "clasificacion": self.resultado_difuso})
                        prev_lado_b = lado_b

            elif self.estado == ESTADO_PLACA:
                cv2.rectangle(frame_display, (10, 10), (450, 50), (0,0,0), -1)
                cv2.putText(frame_display, f"V: {self.velocidad_kmh} km/h - BUSCANDO PLACA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Recortar al area del vehiculo para darle a YOLO una imagen
                # ampliada: la placa ocupa mas porcentaje del frame y es mas facil
                # de detectar que en la escena completa.
                ocr_frame = frame
                ox, oy = 0, 0
                if last_veh_bbox is not None:
                    vbx, vby, vbw, vbh = last_veh_bbox
                    pad_x = int(vbw * 0.5)
                    pad_y = int(vbh * 0.5)
                    rx1 = max(0, vbx - pad_x)
                    ry1 = max(0, vby - pad_y)
                    rx2 = min(frame.shape[1], vbx + vbw + pad_x)
                    ry2 = min(frame.shape[0], vby + vbh + pad_y)
                    ocr_frame = frame[ry1:ry2, rx1:rx2]
                    ox, oy = rx1, ry1
                self.ocr_offset = (ox, oy)
                self.worker.submit(ocr_frame)

                # Acumular TODAS las lecturas nuevas en el votador. NO aceptar la
                # primera: el auto recien cruza la linea y los primeros frames son
                # los mas lejanos/borrosos. Votar entre frames mientras se acerca.
                conf_alta = 0.0
                for p, bb, c in self.worker.drenar_lecturas():
                    self.votador.agregar(p, c)
                    if p:
                        conf_alta = max(conf_alta, c)

                _, bbox, _ = self.worker.get_result()
                if bbox:
                    ox, oy = self.ocr_offset
                    x, y, w, h = bbox
                    cv2.rectangle(frame_display, (x + ox, y + oy), (x + ox + w, y + oy + h), (0, 255, 0), 2)

                # Aceptar el consenso ponderado cuando hay suficientes lecturas
                # (>=5): para entonces el voto por confianza ya es estable.
                placa_aceptada = ""
                consenso = self.votador.consenso()
                if consenso and self.votador.n >= 5:
                    placa_aceptada = consenso
                # Atajo: una lectura muy confiable (auto cerca y nitido) basta.
                elif conf_alta >= 0.80 and consenso:
                    placa_aceptada = consenso

                if placa_aceptada:
                    self.placa_detectada = placa_aceptada
                    self.estado = ESTADO_REGISTRO
                    self.captura_guardada = False
                    self.votador.reset()
                    self._emit("event", {"type": "placa", "placa": self.placa_detectada})

                # Timeout 2.5s: usa lo mejor que haya y libera rapido para no
                # perder los autos siguientes.
                elif time.time() - t_placa_inicio > 2.5:
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
                        t_placa_inicio = 0.0
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
                    t_registro_inicio = time.time()  # inicia el timer de muestra
                    if not self.evento_registrado:
                        registrar_evento(self.placa_detectada, self.velocidad_kmh, clasif, horas, ruta_cap)
                        self.evento_registrado = True
                        datos_correo = {
                            "placa": self.placa_detectada,
                            "velocidad_kmh": self.velocidad_kmh,
                            "clasificacion": clasif,
                            "horas": horas,
                            "ruta_captura": ruta_cap
                        }
                        threading.Thread(target=enviar_notificacion_asincrona, args=(datos_correo,), daemon=True).start()
                        self._emit("event", {"type": "registro_guardado", "placa": self.placa_detectada, "velocidad": self.velocidad_kmh, "clasificacion": clasif, "horas": horas, "captura": ruta_cap})

                cv2.rectangle(frame_display, (10, 10), (500, 170), (0, 0, 0), -1)
                cv2.putText(frame_display, f"PLACA: {self.placa_detectada}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_display, f"VELOCIDAD: {self.velocidad_kmh} km/h", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"ESTADO: {clasif.upper()}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Mostrar 1.5s reales y luego listo para el siguiente vehiculo
                if self.captura_guardada and time.time() - t_registro_inicio > 1.5:
                    self.estado = ESTADO_VELOCIDAD
                    cruzó_linea_a = cruzó_linea_b = False
                    t_a = t_b = 0.0
                    t_placa_inicio = 0.0
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
                target_delay = delay_ms / (1000.0 * getattr(self, 'playback_speed', 1.0))
                sleep_time = target_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

# Instancia global para que el servidor la use
pipeline = RadarPipeline()
