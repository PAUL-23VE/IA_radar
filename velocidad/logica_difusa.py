"""
velocidad/logica_difusa.py
Clasifica la velocidad de un vehículo usando lógica difusa (scikit-fuzzy).

Política del proyecto (parqueadero / zona controlada universitaria):
  0  – 10 km/h  → felicitacion (sin sanción, velocidad prudente)
  10 – 20 km/h  → normal       (velocidad aceptable, sin sanción)
  20 – 30 km/h  → advertencia  (exceso leve, sanción ligera)
  30 – 40 km/h  → multa        (exceso grave, sanción alta)

La salida difusa (`horas`) escala las horas de indisponibilidad según la
gravedad de la infracción. El resultado se formatea en días y horas.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Universo de velocidad (km/h) y de sanción (horas)
V_MAX = 40
H_MAX = 72   # Máximo 3 días (72 horas)

# ── Funciones de membresía de VELOCIDAD (cortes exactos en 10, 20, 30 km/h) ──
#   felicitacion: 0-10 km/h  (pleno 0-8, difuso hasta 12)
#   normal      : 10-20 km/h (centrado en 15)
#   advertencia : 20-30 km/h (centrado en 25)
#   multa       : 30-40 km/h (pleno desde 32)
_MF_FELIZ       = [0,   0,   8,  12]
_MF_NORMAL      = [8,  15,  22]
_MF_ADVERTENCIA = [18, 25,  32]
_MF_MULTA       = [28, 35, V_MAX, V_MAX]


def crear_sistema_difuso():
    """Crea y retorna el simulador del sistema de control difuso (una sola vez)."""
    velocidad = ctrl.Antecedent(np.arange(0, V_MAX + 1, 1), 'velocidad')
    horas     = ctrl.Consequent(np.arange(0, H_MAX + 1, 1), 'horas')

    velocidad['felicitacion'] = fuzz.trapmf(velocidad.universe, _MF_FELIZ)
    velocidad['normal']       = fuzz.trimf(velocidad.universe,  _MF_NORMAL)
    velocidad['advertencia']  = fuzz.trimf(velocidad.universe,  _MF_ADVERTENCIA)
    velocidad['multa']        = fuzz.trapmf(velocidad.universe, _MF_MULTA)

    # Horas de indisponibilidad según gravedad
    horas['ninguna']     = fuzz.trimf(horas.universe, [0,   0,   4])
    horas['leve']        = fuzz.trimf(horas.universe, [2,  12,  24])
    horas['moderada']    = fuzz.trimf(horas.universe, [18, 36,  54])
    horas['severa']      = fuzz.trapmf(horas.universe, [48, 60, H_MAX, H_MAX])

    reglas = [
        ctrl.Rule(velocidad['felicitacion'], horas['ninguna']),
        ctrl.Rule(velocidad['normal'],       horas['ninguna']),
        ctrl.Rule(velocidad['advertencia'],  horas['leve']),
        ctrl.Rule(velocidad['multa'],        horas['severa']),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(reglas))


_sistema = None

def obtener_sistema():
    global _sistema
    if _sistema is None:
        _sistema = crear_sistema_difuso()
    return _sistema


def formatear_tiempo_sancion(horas_total: int) -> str:
    """
    Convierte horas a un formato legible en días y horas.
    Ej: 38 → '1 día con 14 horas'
        24 → '1 día'
        6  → '6 horas'
        0  → 'Sin sanción'
    """
    if horas_total <= 0:
        return "Sin sanción"
    dias = horas_total // 24
    horas_rem = horas_total % 24
    if dias == 0:
        return f"{horas_rem} hora{'s' if horas_rem != 1 else ''}"
    elif horas_rem == 0:
        return f"{dias} día{'s' if dias != 1 else ''}"
    else:
        return f"{dias} día{'s' if dias != 1 else ''} con {horas_rem} hora{'s' if horas_rem != 1 else ''}"


def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna un diccionario con:
      - velocidad:              valor (clamp 0..V_MAX)
      - clasificacion:          'felicitacion', 'normal', 'advertencia' o 'multa'
      - horas_indisponibilidad: horas que el vehículo no puede ingresar
      - tiempo_sancion:         string formateado (ej: '2 días con 4 horas')
      - grados_membresia:       grados difusos de cada categoría
      - mensaje:                descripción para mostrar al usuario
    """
    v = float(np.clip(velocidad_kmh, 0, V_MAX))

    sim = obtener_sistema()
    sim.input['velocidad'] = v
    sim.compute()
    horas_difusas = float(sim.output.get('horas', 0.0))

    # Membresía dominante → clasificación
    universo = np.arange(0, V_MAX + 1, 1)
    grados = {
        'felicitacion': fuzz.interp_membership(universo, fuzz.trapmf(universo, _MF_FELIZ),       v),
        'normal':       fuzz.interp_membership(universo, fuzz.trimf(universo,  _MF_NORMAL),       v),
        'advertencia':  fuzz.interp_membership(universo, fuzz.trimf(universo,  _MF_ADVERTENCIA),  v),
        'multa':        fuzz.interp_membership(universo, fuzz.trapmf(universo, _MF_MULTA),        v),
    }
    clasificacion = max(grados, key=grados.get)

    if clasificacion in ('felicitacion', 'normal'):
        horas = 0
        mensaje = (f"Velocidad prudente ({v:.1f} km/h) — sin sanción"
                   if clasificacion == 'felicitacion'
                   else f"Velocidad normal ({v:.1f} km/h) — sin sanción")
    elif clasificacion == 'advertencia':
        horas = max(1, int(round(horas_difusas)))
        tiempo = formatear_tiempo_sancion(horas)
        mensaje = f"Exceso leve ({v:.1f} km/h) — Advertencia: {tiempo} de indisponibilidad"
    else:  # multa
        horas = max(1, int(round(horas_difusas)))
        tiempo = formatear_tiempo_sancion(horas)
        mensaje = f"EXCESO GRAVE ({v:.1f} km/h) — MULTA: {tiempo} de indisponibilidad"

    return {
        "velocidad":              v,
        "clasificacion":          clasificacion,
        "horas_indisponibilidad": horas,
        "tiempo_sancion":         formatear_tiempo_sancion(horas),
        "grados_membresia":       {k: round(g, 3) for k, g in grados.items()},
        "mensaje":                mensaje,
    }


if __name__ == "__main__":
    casos = [5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 40]
    print("=" * 70)
    print("  SISTEMA DE LÓGICA DIFUSA — Clasificación de velocidad (rangos 10 km/h)")
    print("=" * 70)
    for v in casos:
        r = clasificar_velocidad(v)
        print(f"\n  [{v:3d} km/h] → {r['clasificacion'].upper():12s} | {r['mensaje']}")
        print(f"    Membresía: felicitacion={r['grados_membresia']['felicitacion']:.2f}  "
              f"normal={r['grados_membresia']['normal']:.2f}  "
              f"advertencia={r['grados_membresia']['advertencia']:.2f}  "
              f"multa={r['grados_membresia']['multa']:.2f}")
        if r['horas_indisponibilidad'] > 0:
            print(f"    Sanción: {r['tiempo_sancion']}")
