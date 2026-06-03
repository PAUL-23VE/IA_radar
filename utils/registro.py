"""
utils/registro.py
Registro de eventos en JSON (reemplaza la base de datos PostgreSQL).

Cada evento relevante (placa leída + velocidad + clasificación difusa) se
agrega a registros/eventos.json como una lista de objetos. La ruta de la
captura del fotograma se guarda junto al evento.
"""

import json
import os
from datetime import datetime

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_REGISTROS = os.path.join(ROOT, "registros")
ARCHIVO       = os.path.join(DIR_REGISTROS, "eventos.json")


def _cargar() -> list:
    """Lee la lista de eventos; tolera archivo inexistente o corrupto."""
    if not os.path.exists(ARCHIVO):
        return []
    try:
        with open(ARCHIVO, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def registrar_evento(placa: str, velocidad: float, clasificacion: str,
                     horas: int, ruta_captura: str | None = None) -> dict:
    """
    Agrega un evento al registro JSON y lo retorna.

    Args:
        placa:         placa reconocida (ej. 'ABC-1234')
        velocidad:     km/h medidos
        clasificacion: 'felicitacion' | 'normal' | 'multa'
        horas:         horas de indisponibilidad del vehículo (0 si no hay multa)
        ruta_captura:  ruta de la imagen guardada del fotograma (opcional)
    """
    os.makedirs(DIR_REGISTROS, exist_ok=True)

    evento = {
        "timestamp":               datetime.now().isoformat(timespec="seconds"),
        "placa":                   placa,
        "velocidad_kmh":           round(float(velocidad), 2),
        "clasificacion":           clasificacion,
        "horas_indisponibilidad":  int(horas),
        "ruta_captura":            ruta_captura,
    }

    eventos = _cargar()
    eventos.append(evento)
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(eventos, f, ensure_ascii=False, indent=2)

    return evento


if __name__ == "__main__":
    e = registrar_evento("ABC-1234", 27.5, "multa", 18, "capturas/demo.jpg")
    print(f"Evento registrado en {ARCHIVO}:")
    print(json.dumps(e, ensure_ascii=False, indent=2))
