"""
velocidad/logica_difusa.py
Clasifica la velocidad de un vehículo usando lógica difusa (scikit-fuzzy).

Umbrales del proyecto (del cuaderno):
  v ≤ 20 km/h  → feliz (sin sanción)
  v ~ 25 km/h  → normal
  v ≥ 30 km/h  → multa (días sin ingreso al parqueadero)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ----------------------------------------------------------------
#  DEFINICIÓN DEL SISTEMA DIFUSO
# ----------------------------------------------------------------

def crear_sistema_difuso():
    """
    Crea y retorna el sistema de control difuso configurado.
    Solo se llama una vez.
    """

    # Variable de entrada: velocidad (0 a 60 km/h)
    velocidad = ctrl.Antecedent(np.arange(0, 61, 1), 'velocidad')

    # Variable de salida: nivel de sanción (0 a 10)
    sancion = ctrl.Consequent(np.arange(0, 11, 1), 'sancion')

    # ── Funciones de membresía para VELOCIDAD ────────────────
    #   feliz:  [0, 0, 18, 22]   plenamente feliz hasta 18, difuso hasta 22
    #   normal: [18, 25, 32]     centrado en 25
    #   multa:  [28, 35, 60, 60] multa desde 28, plena desde 35
    velocidad['feliz']  = fuzz.trapmf(velocidad.universe, [0,  0,  18, 22])
    velocidad['normal'] = fuzz.trimf(velocidad.universe,  [18, 25, 32])
    velocidad['multa']  = fuzz.trapmf(velocidad.universe, [28, 35, 60, 60])

    # ── Funciones de membresía para SANCIÓN ──────────────────
    sancion['sin_sancion'] = fuzz.trimf(sancion.universe, [0, 0, 3])
    sancion['leve']        = fuzz.trimf(sancion.universe, [2, 4, 6])
    sancion['severa']      = fuzz.trimf(sancion.universe, [5, 10, 10])

    # ── Reglas difusas ────────────────────────────────────────
    regla1 = ctrl.Rule(velocidad['feliz'],  sancion['sin_sancion'])
    regla2 = ctrl.Rule(velocidad['normal'], sancion['leve'])
    regla3 = ctrl.Rule(velocidad['multa'],  sancion['severa'])

    sistema = ctrl.ControlSystem([regla1, regla2, regla3])
    simulador = ctrl.ControlSystemSimulation(sistema)

    return simulador


# Instancia global (se crea una sola vez)
_sistema = None

def obtener_sistema():
    global _sistema
    if _sistema is None:
        _sistema = crear_sistema_difuso()
    return _sistema


# ----------------------------------------------------------------
#  CLASIFICAR UNA VELOCIDAD
# ----------------------------------------------------------------

def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna un diccionario con:
      - velocidad:      el valor ingresado
      - clasificacion:  'feliz', 'normal' o 'multa'
      - sancion:        valor numérico de sanción (0-10)
      - dias_sin_ingreso: días que no puede ingresar al parqueadero
      - mensaje:        descripción para mostrar al usuario
    """
    # Clamp para evitar errores fuera del universo
    v = float(np.clip(velocidad_kmh, 0, 60))

    sim = obtener_sistema()
    sim.input['velocidad'] = v
    sim.compute()

    nivel_sancion = sim.output['sancion']

    # ── Calcular membresía dominante ──────────────────────────
    universo = np.arange(0, 61, 1)
    mem_feliz  = fuzz.trapmf(universo, [0, 0, 18, 22])
    mem_normal = fuzz.trimf(universo,  [18, 25, 32])
    mem_multa  = fuzz.trapmf(universo, [28, 35, 60, 60])

    idx = int(v)
    grados = {
        'feliz':  fuzz.interp_membership(universo, mem_feliz,  v),
        'normal': fuzz.interp_membership(universo, mem_normal, v),
        'multa':  fuzz.interp_membership(universo, mem_multa,  v)
    }
    clasificacion = max(grados, key=grados.get)

    # ── Días de sanción ───────────────────────────────────────
    if clasificacion == 'feliz':
        dias = 0
        mensaje = f"Velocidad aceptable ({v:.1f} km/h) — Sin sanción"
    elif clasificacion == 'normal':
        dias = int(nivel_sancion * 0.5)          # 0-3 días
        mensaje = f"Velocidad moderada ({v:.1f} km/h) — {dias} día(s) de restricción"
    else:  # multa
        dias = int(nivel_sancion)                 # 0-10 días
        mensaje = f"EXCESO DE VELOCIDAD ({v:.1f} km/h) — {dias} día(s) sin ingreso al parqueadero"

    return {
        "velocidad":        v,
        "clasificacion":    clasificacion,
        "sancion":          round(nivel_sancion, 2),
        "dias_sin_ingreso": dias,
        "grados_membresia": {k: round(g, 3) for k, g in grados.items()},
        "mensaje":          mensaje
    }


# ----------------------------------------------------------------
#  PRUEBA
# ----------------------------------------------------------------

if __name__ == "__main__":
    casos = [10, 18, 22, 25, 28, 32, 40, 55]
    print("\n{'='*60}")
    print("  SISTEMA DE LÓGICA DIFUSA — Clasificación de velocidad")
    print("="*60)
    for v in casos:
        r = clasificar_velocidad(v)
        print(f"\n  {r['mensaje']}")
        print(f"    Membresía: feliz={r['grados_membresia']['feliz']:.2f}  "
              f"normal={r['grados_membresia']['normal']:.2f}  "
              f"multa={r['grados_membresia']['multa']:.2f}")
