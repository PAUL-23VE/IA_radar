"""
utils/reconocedor.py
Módulo para el reconocimiento de placas en segundo plano y votación temporal.
"""

import queue
import threading
from collections import Counter
import numpy as np
from inferencia import reconocer_placa


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
