"""
utils/camara.py
Captura frames desde el iPhone 15 usando HTTP stream.

Opciones para transmitir desde iPhone:
  - EpocCam  (App Store, gratuita con marca de agua, o de pago)
  - DroidCam (también funciona en iOS)
  - IP Camera Lite
  - Camo      (alta calidad)

Una vez instalada la app, ingresar la IP que muestra
en la variable IP_IPHONE de este archivo.
"""

import cv2
import numpy as np
import time


# ----------------------------------------------------------------
#  CONFIGURACIÓN — ajusta la IP según muestre tu app de cámara
# ----------------------------------------------------------------
IP_IPHONE   = "192.168.9.129"       # IP del iPhone en tu red WiFi
PUERTO      = "4747"                 # Puerto por defecto de DroidCam
URL_STREAM  = f"http://{IP_IPHONE}:{PUERTO}/"

# Si prefieres usar la cámara del computador para pruebas:
# URL_STREAM = 0   # 0 = cámara integrada del laptop


# ----------------------------------------------------------------
#  CAPTURA CONTINUA
# ----------------------------------------------------------------

class CamaraIphone:
    """
    Clase para manejar la conexión con el stream del iPhone.
    Uso:
        cam = CamaraIphone()
        frame = cam.obtener_frame()
        cam.liberar()
    """

    def __init__(self, url_base: str = URL_STREAM):
        
        # Lista de rutas comunes según la App (IP Camera, EpocCam, DroidCam, etc.)
        urls_a_probar = [
            url_base,
            f"http://{IP_IPHONE}:{PUERTO}/",
            f"http://{IP_IPHONE}:{PUERTO}/mjpeg",
            f"http://{IP_IPHONE}:{PUERTO}/cam.mjpg",
            f"http://{IP_IPHONE}:{PUERTO}/live"
        ]
        
        print("[Cámara] Intentando conectar con el iPhone...")
        self.cap = None
        url_exitosa = None
        
        for url in urls_a_probar:
            temp_cap = cv2.VideoCapture(url)
            if temp_cap.isOpened():
                # Validar leyendo un frame real
                ret, _ = temp_cap.read()
                if ret:
                    self.cap = temp_cap
                    url_exitosa = url
                    break
            temp_cap.release()

        if self.cap is None:
            raise ConnectionError(
                f"No se pudo conectar al stream en la IP {IP_IPHONE}:{PUERTO}\n"
                "  Verifica que:\n"
                "  1. El iPhone y el computador están en la misma red WiFi\n"
                "  2. La app está transmitiendo video en este momento\n"
            )

        print(f"[Cámara] Conexión establecida ✓ (Ruta: {url_exitosa})")

    def obtener_frame(self) -> np.ndarray | None:
        """Retorna el frame actual o None si falló."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def liberar(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.liberar()


# ----------------------------------------------------------------
#  FUNCIÓN PARA CAPTURAR FOTO EN EL MOMENTO CORRECTO
# ----------------------------------------------------------------

def capturar_frame_con_vehiculo(url: str = URL_STREAM,
                                 timeout_seg: int = 30) -> np.ndarray | None:
    """
    Muestra el stream en vivo y captura el frame cuando el usuario
    presiona ESPACIO, o devuelve None si pasa el timeout.

    En producción esto sería automático (detección de movimiento).
    """
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("[Error] No se pudo abrir el stream.")
        return None

    print("\nStream abierto. Presiona ESPACIO para capturar, Q para salir.")
    inicio = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mostrar preview
        cv2.putText(frame, "ESPACIO = capturar | Q = salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("Stream iPhone", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(' '):           # ESPACIO → capturar
            cap.release()
            cv2.destroyAllWindows()
            print("[Cámara] Frame capturado.")
            return frame

        if tecla == ord('q'):           # Q → salir
            break

        if time.time() - inicio > timeout_seg:
            print("[Cámara] Timeout alcanzado.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


# ----------------------------------------------------------------
#  TEST
# ----------------------------------------------------------------

if __name__ == "__main__":
    # Probar con la cámara del laptop (índice 0) si no hay iPhone
    try:
        with CamaraIphone(url=0) as cam:
            for _ in range(5):
                f = cam.obtener_frame()
                if f is not None:
                    cv2.imshow("Test", f)
                    cv2.waitKey(300)
            cv2.destroyAllWindows()
            print("Cámara funciona correctamente.")
    except Exception as e:
        print(f"Error: {e}")
