"""
main.py
Pipeline completo del sistema de reconocimiento de placas.

Flujo:
  1. Captura frame desde iPhone
  2. CNN detecta y lee la placa → string "ABC-1234"
  3. Consulta PostgreSQL → datos del vehículo y propietario
  4. Mide velocidad con dos frames + lógica difusa
  5. Si hay multa → registra en BD y notifica

Ejecutar: python main.py
"""

import cv2
import numpy as np
import time
import sys
import os

# Agregar rutas de submódulos al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'velocidad'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from inferencia      import reconocer_placa
from db_connection   import buscar_vehiculo, registrar_multa, imprimir_vehiculo
from logica_difusa   import clasificar_velocidad
from camara          import CamaraIphone, URL_STREAM

# ----------------------------------------------------------------
#  DISTANCIA DE REFERENCIA PARA MEDIR VELOCIDAD
#  Mide físicamente cuántos metros hay entre los dos puntos
#  en el campo visual de la cámara.
# ----------------------------------------------------------------
DISTANCIA_REFERENCIA_METROS = 5.0    # metros entre línea A y línea B


# ----------------------------------------------------------------
#  MÁQUINA DE ESTADOS (STATE MACHINE) Y PIPELINE PRINCIPAL
# ----------------------------------------------------------------
ESTADO_VELOCIDAD = 0
ESTADO_PLACA = 1
ESTADO_BD = 2

def procesar_vehiculo(cam_url: str = URL_STREAM, distancia_m: float = DISTANCIA_REFERENCIA_METROS):
    """
    Pipeline principal concurrente. Procesa velocidad y placa en el mismo frame.
    Dibuja un HUD integrado.
    """
    print("\n" + "="*60)
    print("  SISTEMA INTEGRADO DE PLACAS — UTA")
    print("="*60)

    cap = None
    if isinstance(cam_url, str) and cam_url.startswith("http"):
        # Extraer base (quitar /video, / etc si existe)
        base = cam_url.rstrip("/")
        if base.endswith("/video"): base = base[:-6]
        
        rutas_a_probar = [
            f"{base}/video",
            f"{base}/",
            f"{base}/mjpeg",
            f"{base}/live"
        ]
        print("[Sistema] Buscando señal de video en el teléfono...")
        for ruta in rutas_a_probar:
            temp_cap = cv2.VideoCapture(ruta)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    cap = temp_cap
                    print(f"[Sistema] ¡Cámara conectada! ({ruta})")
                    break
            temp_cap.release()
    else:
        # Cámara local u otra ruta
        cap = cv2.VideoCapture(cam_url)

    if cap is None or not cap.isOpened():
        print("[ERROR] No se pudo abrir el stream de la cámara del teléfono.")
        print("Asegúrate de que la app esté abierta y transmitiendo en la misma red WiFi.")
        return

    # Variables de estado
    estado = ESTADO_VELOCIDAD
    
    # Tracker de velocidad
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    t_a = 0.0
    t_b = 0.0
    cruzó_linea_a = False
    cruzó_linea_b = False
    velocidad_kmh = 0.0
    resultado_difuso = None
    
    # Placa y BD
    placa_detectada = ""
    datos_vehiculo = None
    
    ret, frame_init = cap.read()
    if not ret: return
    alto, ancho = frame_init.shape[:2]
    linea_a_y = int(alto * 0.3)
    linea_b_y = int(alto * 0.7)

    print("[Sistema] Iniciando monitoreo... (Presiona ESC para salir, R para reiniciar)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_display = frame.copy()
            
            # ─── MÓDULO VELOCIDAD (Concurrente) ───────────────────
            if estado == ESTADO_VELOCIDAD:
                mascara = fgbg.apply(frame)
                _, mascara_bin = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mascara_limpia = cv2.morphologyEx(mascara_bin, cv2.MORPH_OPEN, kernel)
                mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)
                contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                vehiculo_y = None
                for cnt in contornos:
                    if cv2.contourArea(cnt) > 3000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        vehiculo_y = y + h
                        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.circle(frame_display, (x + int(w/2), vehiculo_y), 5, (0, 0, 255), -1)
                        break
                        
                cv2.line(frame_display, (0, linea_a_y), (ancho, linea_a_y), (255, 0, 0), 2)
                cv2.putText(frame_display, "Linea A", (10, linea_a_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.line(frame_display, (0, linea_b_y), (ancho, linea_b_y), (0, 0, 255), 2)
                cv2.putText(frame_display, "Linea B", (10, linea_b_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # HUD Velocidad
                cv2.putText(frame_display, "ESTADO: MIDIENDO VELOCIDAD...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if vehiculo_y is not None:
                    if not cruzó_linea_a and vehiculo_y > linea_a_y:
                        t_a = time.time()
                        cruzó_linea_a = True
                        print("[Velocidad] Vehículo cruzó Línea A")
                    if cruzó_linea_a and not cruzó_linea_b and vehiculo_y > linea_b_y:
                        t_b = time.time()
                        cruzó_linea_b = True
                        dt = t_b - t_a
                        if dt > 0:
                            velocidad_kmh = (distancia_m / dt) * 3.6
                            velocidad_kmh = round(velocidad_kmh, 2)
                        resultado_difuso = clasificar_velocidad(velocidad_kmh)
                        estado = ESTADO_PLACA
                        print(f"[Velocidad] Completado: {velocidad_kmh} km/h")
                        print(f"[Sistema] Cambiando a búsqueda de Placa...")
            
            # ─── MÓDULO PLACA (Concurrente tras velocidad) ───────
            elif estado == ESTADO_PLACA:
                cv2.putText(frame_display, f"V: {velocidad_kmh} km/h - {resultado_difuso['clasificacion'].upper()}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame_display, "ESTADO: BUSCANDO PLACA...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                placa, bbox, conf = reconocer_placa(frame)
                if placa and conf >= 0.50:
                    x, y, w, h = bbox
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    placa_detectada = placa
                    estado = ESTADO_BD
                    print(f"[Placa] Reconocida: {placa} (Conf: {conf:.2f})")
                    print(f"[Sistema] Consultando Base de Datos...")
                elif placa:
                    print(f"[Placa] Descartada por baja confianza: {placa} (Conf: {conf:.2f})")
                
                # Timeout para evitar que el sistema se quede congelado si nunca detecta la placa
                if time.time() - t_b > 5.0:
                    print("[Sistema] Tiempo agotado buscando placa. Reiniciando radar...")
                    estado = ESTADO_VELOCIDAD
                    cruzó_linea_a = False
                    cruzó_linea_b = False
                    t_a = 0.0
                    t_b = 0.0


            # ─── MÓDULO DB Y HUD FINAL ───────────────────────────
            elif estado == ESTADO_BD:
                if datos_vehiculo is None:
                    datos_vehiculo = buscar_vehiculo(placa_detectada)
                    if resultado_difuso['clasificacion'] in ('normal', 'multa'):
                        registrar_multa(placa_detectada, velocidad_kmh,
                                        resultado_difuso['clasificacion'],
                                        resultado_difuso['dias_sin_ingreso'])
                
                # DIBUJAR HUD COMPLETO
                cv2.rectangle(frame_display, (10, 10), (500, 160), (0, 0, 0), -1)
                cv2.putText(frame_display, f"PLACA: {placa_detectada}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_display, f"VELOCIDAD: {velocidad_kmh} km/h", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                color_difuso = (0, 255, 0)
                if resultado_difuso['clasificacion'] == 'multa': color_difuso = (0, 0, 255)
                elif resultado_difuso['clasificacion'] == 'normal': color_difuso = (0, 255, 255)
                
                cv2.putText(frame_display, f"ESTADO: {resultado_difuso['clasificacion'].upper()}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_difuso, 2)
                
                if datos_vehiculo:
                    cv2.putText(frame_display, f"INFO: {datos_vehiculo['marca']} {datos_vehiculo['modelo']}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame_display, "Presiona 'R' para reiniciar o 'ESC' para salir", (10, alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Sistema Integrado de Placas", frame_display)
            
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == 27: # ESC
                break
            elif tecla == ord('r') or tecla == ord('R'):
                estado = ESTADO_VELOCIDAD
                cruzó_linea_a = False
                cruzó_linea_b = False
                t_a = 0.0
                t_b = 0.0
                placa_detectada = ""
                datos_vehiculo = None
                resultado_difuso = None
                velocidad_kmh = 0.0
                print("\n[Sistema] Reiniciando...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------
#  MODO DEMO (sin cámara real — usa imagen estática)
# ----------------------------------------------------------------

def demo_sin_camara(placa_prueba: str = "ABC-1234",
                     velocidad_prueba: float = 35.0):
    """
    Demo completo sin necesitar la cámara ni el modelo entrenado.
    Solo prueba la conexión BD + lógica difusa.
    """
    print("\n" + "="*60)
    print("  MODO DEMO — sin cámara ni CNN")
    print("="*60)

    # Simular velocidad
    print(f"\n[Demo] Velocidad simulada: {velocidad_prueba} km/h")
    resultado = clasificar_velocidad(velocidad_prueba)
    print(resultado['mensaje'])
    print(f"  Grados de membresía: {resultado['grados_membresia']}")

    # Buscar en BD
    print(f"\n[Demo] Buscando placa: {placa_prueba}")
    datos = buscar_vehiculo(placa_prueba)
    imprimir_vehiculo(datos)

    # Registrar multa si aplica
    if resultado['clasificacion'] in ('normal', 'multa') and datos:
        registrar_multa(placa_prueba, velocidad_prueba,
                        resultado['clasificacion'],
                        resultado['dias_sin_ingreso'])
        print("✅ Multa registrada en BD.")


# ----------------------------------------------------------------
#  ENTRADA
# ----------------------------------------------------------------

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if modo == "demo":
        # Probar solo BD + lógica difusa (no necesita cámara ni CNN)
        demo_sin_camara("ABC-1234", 35.0)
        demo_sin_camara("XYZ-4567", 18.0)
        demo_sin_camara("KLM-1234", 25.0)

    elif modo == "live":
        # Pipeline completo con cámara real
        procesar_vehiculo(cam_url=URL_STREAM)

    elif modo == "laptop":
        # Pipeline con cámara del laptop (para pruebas)
        procesar_vehiculo(cam_url=0)
