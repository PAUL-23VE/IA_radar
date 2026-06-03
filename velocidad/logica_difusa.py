"""
velocidad/logica_difusa.py
Clasifica la velocidad de un vehículo usando lógica difusa (scikit-fuzzy).

Política del proyecto (parqueadero / zona controlada):
  0 – 10 km/h   → felicitaciones (sin sanción)
  10 – 20 km/h  → normal (advertencia, sin sanción)
  > 20 km/h     → MULTA: X horas de indisponibilidad del vehículo

La salida difusa (`horas`) escala las horas de indisponibilidad según el
exceso de velocidad por encima de 20 km/h.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Universo de velocidad (km/h) y de sanción (horas)
V_MAX = 40
H_MAX = 48

# ── Funciones de membresía de VELOCIDAD (cortes en 10 y 20 km/h) ──
#   felicitacion: pleno hasta 8, difuso hasta 11
#   normal      : centrado en 15 (zona 10-20)
#   multa se subdivide para que las HORAS escalen con el exceso:
#     multa_leve : exceso moderado (~20-30 km/h)
#     multa_grave: exceso fuerte   (> 28 km/h)
_MF_FELIZ       = [0,  0,  8,  11]
_MF_NORMAL      = [9,  15, 21]
_MF_MULTA_LEVE  = [19, 24, 30]
_MF_MULTA_GRAVE = [28, 34, V_MAX, V_MAX]


def crear_sistema_difuso():
    """Crea y retorna el simulador del sistema de control difuso (una sola vez)."""
    velocidad = ctrl.Antecedent(np.arange(0, V_MAX + 1, 1), 'velocidad')
    horas     = ctrl.Consequent(np.arange(0, H_MAX + 1, 1), 'horas')

    velocidad['felicitacion'] = fuzz.trapmf(velocidad.universe, _MF_FELIZ)
    velocidad['normal']       = fuzz.trimf(velocidad.universe,  _MF_NORMAL)
    velocidad['multa_leve']   = fuzz.trimf(velocidad.universe,  _MF_MULTA_LEVE)
    velocidad['multa_grave']  = fuzz.trapmf(velocidad.universe, _MF_MULTA_GRAVE)

    # Horas de indisponibilidad: ninguna / moderada / alta
    horas['ninguna']  = fuzz.trimf(horas.universe, [0, 0, 6])
    horas['moderada'] = fuzz.trimf(horas.universe, [4, 16, 28])
    horas['alta']     = fuzz.trapmf(horas.universe, [24, 40, H_MAX, H_MAX])

    reglas = [
        ctrl.Rule(velocidad['felicitacion'], horas['ninguna']),
        ctrl.Rule(velocidad['normal'],       horas['ninguna']),
        ctrl.Rule(velocidad['multa_leve'],   horas['moderada']),
        ctrl.Rule(velocidad['multa_grave'],  horas['alta']),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(reglas))


_sistema = None

def obtener_sistema():
    global _sistema
    if _sistema is None:
        _sistema = crear_sistema_difuso()
    return _sistema


def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna un diccionario con:
      - velocidad:              valor (clamp 0..V_MAX)
      - clasificacion:          'felicitacion', 'normal' o 'multa'
      - horas_indisponibilidad: horas que el vehículo no puede ingresar (solo multa)
      - grados_membresia:       grados difusos de cada categoría
      - mensaje:                descripción para mostrar al usuario
    """
    v = float(np.clip(velocidad_kmh, 0, V_MAX))

    sim = obtener_sistema()
    sim.input['velocidad'] = v
    sim.compute()
    horas_difusas = float(sim.output.get('horas', 0.0))

    # Membresía dominante → clasificación (multa = max de leve/grave)
    universo = np.arange(0, V_MAX + 1, 1)
    g_leve  = fuzz.interp_membership(universo, fuzz.trimf(universo,  _MF_MULTA_LEVE),  v)
    g_grave = fuzz.interp_membership(universo, fuzz.trapmf(universo, _MF_MULTA_GRAVE), v)
    grados = {
        'felicitacion': fuzz.interp_membership(universo, fuzz.trapmf(universo, _MF_FELIZ),  v),
        'normal':       fuzz.interp_membership(universo, fuzz.trimf(universo,  _MF_NORMAL), v),
        'multa':        max(g_leve, g_grave),
    }
    clasificacion = max(grados, key=grados.get)

    if clasificacion == 'felicitacion':
        horas = 0
        mensaje = f"Felicitaciones — velocidad prudente ({v:.1f} km/h)"
    elif clasificacion == 'normal':
        horas = 0
        mensaje = f"Velocidad normal ({v:.1f} km/h) — sin sanción"
    else:  # multa
        horas = int(round(horas_difusas))
        mensaje = (f"EXCESO DE VELOCIDAD ({v:.1f} km/h) — "
                   f"{horas} hora(s) de indisponibilidad del vehículo")

    return {
        "velocidad":              v,
        "clasificacion":          clasificacion,
        "horas_indisponibilidad": horas,
        "grados_membresia":       {k: round(g, 3) for k, g in grados.items()},
        "mensaje":                mensaje,
    }


if __name__ == "__main__":
    casos = [5, 8, 12, 15, 18, 22, 28, 35, 40]
    print("=" * 64)
    print("  SISTEMA DE LÓGICA DIFUSA — Clasificación de velocidad")
    print("=" * 64)
    for v in casos:
        r = clasificar_velocidad(v)
        print(f"\n  {r['mensaje']}")
        print(f"    Membresía: felicitacion={r['grados_membresia']['felicitacion']:.2f}  "
              f"normal={r['grados_membresia']['normal']:.2f}  "
              f"multa={r['grados_membresia']['multa']:.2f}")
