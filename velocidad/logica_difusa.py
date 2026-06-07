"""
velocidad/logica_difusa.py
Clasifica la velocidad de un vehículo usando lógica difusa (scikit-fuzzy).

Política del proyecto (parqueadero / zona controlada universitaria):
  0  – 10 km/h  → felicitacion (velocidad prudente, sin sanción)
  10 – 20 km/h  → normal       (velocidad aceptable, sin sanción)
  > 20 km/h     → MULTA        (X horas de indisponibilidad del vehículo)

El sistema difuso entrega SOLO tres resultados: felicitacion, normal o multa.
Cuando es multa, la salida difusa (`horas`) escala suavemente con el exceso de
velocidad por encima de 20 km/h: a más velocidad, más horas de indisponibilidad.
El resultado se formatea en días y horas.
"""

import io
import base64

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Universo de velocidad (km/h) y de sanción (horas)
V_MAX = 40
H_MAX = 72   # Máximo 3 días (72 horas)

# ── Funciones de membresía de VELOCIDAD (cortes en 10 y 20 km/h) ──
#   felicitacion: 0-10 km/h  (pleno 0-8, difuso hasta 11)
#   normal      : 10-20 km/h (centrado en 15)
#   multa (>20) : se sub-divide SOLO para que las HORAS escalen con el exceso:
#       multa_leve  ~20-30 km/h  → sanción baja
#       multa_media ~27-39 km/h  → sanción media
#       multa_grave ~35-40 km/h  → sanción alta
#   La clasificación de display es una sola: 'multa' = max(leve, media, grave).
_MF_FELIZ        = [0,   0,   8,  11]
_MF_NORMAL       = [9,  15,  21]
_MF_MULTA_LEVE   = [19, 25,  31]
_MF_MULTA_MEDIA  = [27, 33,  39]
_MF_MULTA_GRAVE  = [35, 39, V_MAX, V_MAX]

# ── Funciones de membresía de HORAS de indisponibilidad ──
_MF_H_NINGUNA = [0,   0,   4]
_MF_H_BAJA    = [3,  12,  22]
_MF_H_MEDIA   = [18, 32,  48]
_MF_H_ALTA    = [42, 56, H_MAX, H_MAX]


def crear_sistema_difuso():
    """Crea y retorna el simulador del sistema de control difuso (una sola vez)."""
    velocidad = ctrl.Antecedent(np.arange(0, V_MAX + 1, 1), 'velocidad')
    horas     = ctrl.Consequent(np.arange(0, H_MAX + 1, 1), 'horas')

    velocidad['felicitacion'] = fuzz.trapmf(velocidad.universe, _MF_FELIZ)
    velocidad['normal']       = fuzz.trimf(velocidad.universe,  _MF_NORMAL)
    velocidad['multa_leve']   = fuzz.trimf(velocidad.universe,  _MF_MULTA_LEVE)
    velocidad['multa_media']  = fuzz.trimf(velocidad.universe,  _MF_MULTA_MEDIA)
    velocidad['multa_grave']  = fuzz.trapmf(velocidad.universe, _MF_MULTA_GRAVE)

    horas['ninguna'] = fuzz.trimf(horas.universe,  _MF_H_NINGUNA)
    horas['baja']    = fuzz.trimf(horas.universe,  _MF_H_BAJA)
    horas['media']   = fuzz.trimf(horas.universe,  _MF_H_MEDIA)
    horas['alta']    = fuzz.trapmf(horas.universe, _MF_H_ALTA)

    reglas = [
        ctrl.Rule(velocidad['felicitacion'], horas['ninguna']),
        ctrl.Rule(velocidad['normal'],       horas['ninguna']),
        ctrl.Rule(velocidad['multa_leve'],   horas['baja']),
        ctrl.Rule(velocidad['multa_media'],  horas['media']),
        ctrl.Rule(velocidad['multa_grave'],  horas['alta']),
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


def _grados_velocidad(v: float) -> dict:
    """Grados de membresía de las 3 categorías de DISPLAY en la velocidad v."""
    u = np.arange(0, V_MAX + 1, 1)
    g_leve  = fuzz.interp_membership(u, fuzz.trimf(u,  _MF_MULTA_LEVE),  v)
    g_media = fuzz.interp_membership(u, fuzz.trimf(u,  _MF_MULTA_MEDIA), v)
    g_grave = fuzz.interp_membership(u, fuzz.trapmf(u, _MF_MULTA_GRAVE), v)
    return {
        'felicitacion': float(fuzz.interp_membership(u, fuzz.trapmf(u, _MF_FELIZ),  v)),
        'normal':       float(fuzz.interp_membership(u, fuzz.trimf(u,  _MF_NORMAL), v)),
        'multa':        float(max(g_leve, g_media, g_grave)),
    }


def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna un diccionario con:
      - velocidad:              valor (clamp 0..V_MAX)
      - clasificacion:          'felicitacion', 'normal' o 'multa'
      - horas_indisponibilidad: horas que el vehículo no puede ingresar (solo multa)
      - tiempo_sancion:         string formateado (ej: '2 días con 4 horas')
      - grados_membresia:       grados difusos de cada categoría
      - mensaje:                descripción para mostrar al usuario
    """
    v = float(np.clip(velocidad_kmh, 0, V_MAX))

    sim = obtener_sistema()
    sim.input['velocidad'] = v
    sim.compute()
    horas_difusas = float(sim.output.get('horas', 0.0))

    grados = _grados_velocidad(v)
    clasificacion = max(grados, key=grados.get)

    if clasificacion == 'felicitacion':
        horas = 0
        mensaje = f"Felicitaciones — velocidad prudente ({v:.1f} km/h), sin sanción"
    elif clasificacion == 'normal':
        horas = 0
        mensaje = f"Velocidad normal ({v:.1f} km/h) — sin sanción"
    else:  # multa
        horas = max(1, int(round(horas_difusas)))
        tiempo = formatear_tiempo_sancion(horas)
        mensaje = f"MULTA por exceso de velocidad ({v:.1f} km/h) — {tiempo} de indisponibilidad"

    return {
        "velocidad":              v,
        "clasificacion":          clasificacion,
        "horas_indisponibilidad": horas,
        "tiempo_sancion":         formatear_tiempo_sancion(horas),
        "grados_membresia":       {k: round(g, 3) for k, g in grados.items()},
        "mensaje":                mensaje,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualización del razonamiento difuso (Mamdani) para mostrar en UI / correo
# ──────────────────────────────────────────────────────────────────────────────
def _render_inferencia_png(velocidad_kmh: float) -> bytes:
    """
    Genera la figura del razonamiento difuso (Mamdani) para una velocidad dada
    y retorna los bytes PNG.

    Panel superior  : funciones de membresía de VELOCIDAD (felicitacion/normal/
                      multa) con línea vertical en la velocidad medida y los
                      grados de pertenencia marcados.
    Panel inferior  : funciones de membresía de HORAS, el área agregada
                      (recortada por las reglas) y el centroide = defuzzificación.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = float(np.clip(velocidad_kmh, 0, V_MAX))
    res = clasificar_velocidad(v)

    vu = np.arange(0, V_MAX + 0.5, 0.5)
    hu = np.arange(0, H_MAX + 0.5, 0.5)

    # Membresías de velocidad
    mf_feliz  = fuzz.trapmf(vu, _MF_FELIZ)
    mf_normal = fuzz.trimf(vu,  _MF_NORMAL)
    mf_leve   = fuzz.trimf(vu,  _MF_MULTA_LEVE)
    mf_media  = fuzz.trimf(vu,  _MF_MULTA_MEDIA)
    mf_grave  = fuzz.trapmf(vu, _MF_MULTA_GRAVE)
    mf_multa  = np.maximum.reduce([mf_leve, mf_media, mf_grave])

    # Membresías de horas
    h_ninguna = fuzz.trimf(hu,  _MF_H_NINGUNA)
    h_baja    = fuzz.trimf(hu,  _MF_H_BAJA)
    h_media   = fuzz.trimf(hu,  _MF_H_MEDIA)
    h_alta    = fuzz.trapmf(hu, _MF_H_ALTA)

    # Activaciones de reglas a la velocidad v
    a_feliz = fuzz.interp_membership(vu, mf_feliz,  v)
    a_norm  = fuzz.interp_membership(vu, mf_normal, v)
    a_leve  = fuzz.interp_membership(vu, mf_leve,   v)
    a_media = fuzz.interp_membership(vu, mf_media,  v)
    a_grave = fuzz.interp_membership(vu, mf_grave,  v)

    act_ninguna = max(a_feliz, a_norm)
    # Salida recortada (implicación de Mamdani por mínimo) y agregada por máximo
    agg = np.maximum.reduce([
        np.minimum(act_ninguna, h_ninguna),
        np.minimum(a_leve,      h_baja),
        np.minimum(a_media,     h_media),
        np.minimum(a_grave,     h_alta),
    ])
    try:
        centroide = fuzz.defuzz(hu, agg, 'centroid') if agg.max() > 0 else 0.0
    except Exception:
        centroide = float(res["horas_indisponibilidad"])

    C_FELIZ, C_NORMAL, C_MULTA = "#10b981", "#3b82f6", "#ef4444"
    C_OUT = "#8b5cf6"

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), dpi=110)
    fig.patch.set_facecolor("white")

    # ── Panel 1: VELOCIDAD ──
    ax1.plot(vu, mf_feliz,  color=C_FELIZ,  lw=2, label="Felicitación (0–10)")
    ax1.fill_between(vu, mf_feliz,  alpha=0.12, color=C_FELIZ)
    ax1.plot(vu, mf_normal, color=C_NORMAL, lw=2, label="Normal (10–20)")
    ax1.fill_between(vu, mf_normal, alpha=0.12, color=C_NORMAL)
    ax1.plot(vu, mf_multa,  color=C_MULTA,  lw=2, label="Multa (>20)")
    ax1.fill_between(vu, mf_multa,  alpha=0.12, color=C_MULTA)

    ax1.axvline(v, color="#111827", ls="--", lw=1.6)
    for grado, color in ((res["grados_membresia"]["felicitacion"], C_FELIZ),
                         (res["grados_membresia"]["normal"], C_NORMAL),
                         (res["grados_membresia"]["multa"], C_MULTA)):
        if grado > 0.01:
            ax1.plot(v, grado, "o", color=color, ms=8, mec="white", mew=1.2, zorder=5)
    ax1.text(v, 1.06, f"{v:.1f} km/h", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#111827")
    ax1.set_title("Entrada difusa — Velocidad", fontsize=11, fontweight="bold", loc="left")
    ax1.set_xlabel("km/h", fontsize=9)
    ax1.set_ylabel("Pertenencia", fontsize=9)
    ax1.set_ylim(0, 1.18)
    ax1.set_xlim(0, V_MAX)
    ax1.legend(fontsize=8, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    ax1.grid(alpha=0.15)

    # ── Panel 2: HORAS (salida) ──
    for mf, name, col in ((h_ninguna, "Ninguna", "#9ca3af"), (h_baja, "Baja", "#fbbf24"),
                          (h_media, "Media", "#fb923c"), (h_alta, "Alta", "#dc2626")):
        ax2.plot(hu, mf, color=col, lw=1.3, ls=":", label=name)
    ax2.fill_between(hu, agg, color=C_OUT, alpha=0.35, label="Agregado")
    if agg.max() > 0:
        ax2.axvline(centroide, color=C_OUT, lw=2)
        ax2.text(centroide, 1.06, f"{res['horas_indisponibilidad']} h",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_OUT)
    ax2.set_title("Salida difusa — Horas de indisponibilidad (centroide)",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlabel("horas", fontsize=9)
    ax2.set_ylabel("Pertenencia", fontsize=9)
    ax2.set_ylim(0, 1.18)
    ax2.set_xlim(0, H_MAX)
    ax2.legend(fontsize=8, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    ax2.grid(alpha=0.15)

    clasif = res["clasificacion"].upper()
    color_t = {"FELICITACION": C_FELIZ, "NORMAL": C_NORMAL, "MULTA": C_MULTA}.get(clasif, "#111")
    fig.suptitle(f"Resultado: {clasif}  ·  {res['tiempo_sancion']}",
                 fontsize=12, fontweight="bold", color=color_t)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def graficar_inferencia(velocidad_kmh: float) -> str:
    """Razonamiento difuso como data-URI PNG base64 (para <img> en la UI/modal)."""
    png = _render_inferencia_png(velocidad_kmh)
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def guardar_inferencia_png(velocidad_kmh: float, ruta: str) -> str:
    """Razonamiento difuso guardado como archivo PNG (para adjuntar inline en el correo)."""
    png = _render_inferencia_png(velocidad_kmh)
    with open(ruta, "wb") as f:
        f.write(png)
    return ruta


if __name__ == "__main__":
    casos = [5, 8, 12, 15, 18, 21, 24, 28, 32, 35, 38, 40]
    print("=" * 72)
    print("  SISTEMA DE LÓGICA DIFUSA — felicitacion / normal / multa (>20 km/h)")
    print("=" * 72)
    for v in casos:
        r = clasificar_velocidad(v)
        print(f"\n  [{v:3d} km/h] → {r['clasificacion'].upper():12s} | {r['mensaje']}")
        print(f"    Membresía: felicitacion={r['grados_membresia']['felicitacion']:.2f}  "
              f"normal={r['grados_membresia']['normal']:.2f}  "
              f"multa={r['grados_membresia']['multa']:.2f}")
        if r['horas_indisponibilidad'] > 0:
            print(f"    Sanción: {r['tiempo_sancion']} ({r['horas_indisponibilidad']} h)")
