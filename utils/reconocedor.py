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

class RecognitionWorker:
    """
    Procesa frames en un hilo separado para no bloquear la captura.
    Solo mantiene el frame más reciente en la cola (descarta frames viejos).
    Cada lectura completada se acumula para que el loop principal pueda votarlas.
    """

    def __init__(self):
        self._inbox   = queue.Queue(maxsize=1)
        self._result  = ("", None, 0.0)   # última lectura (para el bbox/HUD)
        self._pendientes = []             # lecturas nuevas no consumidas
        self._lock    = threading.Lock()
        self._active  = False
        self.thread = None

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

    def start(self) -> None:
        if not self._active:
            self._active = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self._active = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

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

    def __init__(self, min_votos: int = 3, conf_min: float = 0.35):
        self.min_votos = min_votos
        self.conf_min  = conf_min
        self._lecturas: list[str] = []
        self._confs: list[float] = []

    def agregar(self, placa: str, conf: float) -> None:
        if placa and conf >= self.conf_min:
            self._lecturas.append(placa)
            self._confs.append(conf)

    def votos_consenso(self) -> int:
        """Cuantas lecturas coinciden con el consenso actual (estabilidad)."""
        c = self.consenso()
        if not c:
            return 0
        return sum(1 for p in self._lecturas if p == c)

    @property
    def n(self) -> int:
        return len(self._lecturas)

    def mejor_lectura(self) -> tuple[str, float]:
        """Retorna (placa, conf) de la lectura con mayor confianza acumulada."""
        if not self._lecturas:
            return "", 0.0
        idx = max(range(len(self._confs)), key=lambda i: self._confs[i])
        return self._lecturas[idx], self._confs[idx]

    def consenso(self) -> str:
        """
        Consenso por posición PONDERADO por confianza. Los frames más nítidos
        (mayor conf, auto más cerca) pesan más en cada carácter. Esto estabiliza
        las letras que bailan entre frames borrosos por motion blur.
        '' si no hay suficientes votos.
        """
        if len(self._lecturas) < self.min_votos:
            return ""
        # Agrupar por longitud (ABC-NNN vs ABC-NNNN) y usar la más frecuente
        longitud = Counter(len(p) for p in self._lecturas).most_common(1)[0][0]
        grupo    = [(p, c) for p, c in zip(self._lecturas, self._confs)
                    if len(p) == longitud]
        # Por cada posición, suma de confianza por carácter → gana el de mayor masa
        res = []
        for i in range(longitud):
            masa: dict[str, float] = {}
            for p, c in grupo:
                masa[p[i]] = masa.get(p[i], 0.0) + c
            res.append(max(masa, key=masa.get))
        return "".join(res)

    def reset(self) -> None:
        self._lecturas = []
        self._confs = []
