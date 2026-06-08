"""
velocidad/logica_difusa.py
Clasifica la velocidad de un vehículo con lógica difusa y calcula la sanción.

Política del proyecto (parqueadero / zona controlada universitaria, límite 20 km/h):
  0  – 10 km/h  → felicitacion (velocidad prudente, sin sanción)
  10 – 20 km/h  → normal       (velocidad aceptable, sin sanción)
  > 20 km/h     → MULTA        (horas de indisponibilidad PROPORCIONALES al exceso)

Diseño en dos etapas:
  1. CLASIFICACIÓN (Mamdani de pertenencias): tres conjuntos difusos de velocidad
     —felicitacion, normal, multa—; la categoría es la de mayor pertenencia.
     La pertenencia 'multa' es una RAMPA monótona desde 20 km/h (no un serrucho).
  2. SANCIÓN (Takagi–Sugeno de orden cero): SÓLO cuando la categoría es multa, las
     horas se obtienen como promedio ponderado de cinco sub-niveles de severidad
     (leve…extrema) con consecuentes singleton crecientes. Esto produce una salida
     SUAVE y ESTRICTAMENTE MONÓTONA: a más exceso sobre 20 km/h, más horas.
     (Sugeno se elige sobre el centroide de Mamdani porque éste introduce mesetas
     y micro-descensos no monótonos cerca de los bordes de cada conjunto.)
"""

import io
import base64

import numpy as np
import skfuzzy as fuzz


# Universo de velocidad (km/h) y máximo de sanción (horas)
V_MAX = 40
H_MAX = 72   # 3 días

# ── Pertenencias de VELOCIDAD (clasificación: 3 categorías) ──
#   La multa es una RAMPA limpia y monótona desde 20 km/h (antes era el máximo de
#   tres triángulos solapados -> aparecía como serrucho en la gráfica).
_MF_FELIZ  = [0,  0,  8, 11]   # trapezoidal
_MF_NORMAL = [9, 15, 21]       # triangular
_MF_MULTA  = [20, 25, V_MAX, V_MAX]   # trapezoidal (rampa 20->25, plena hasta 40)

# ── Sub-niveles de severidad de la multa (Takagi–Sugeno) ──
# Triángulos que teselan 20–40 al 50% de solape; cada uno tiene un consecuente
# singleton (horas). El promedio ponderado por pertenencia da la sanción.
_SUBNIVELES = [
    ("leve",     [20, 24, 28],          5),    # ~exceso 1-8  km/h
    ("moderada", [24, 28, 32],          18),   # ~exceso 5-12 km/h
    ("alta",     [28, 32, 36],          34),   # ~exceso 9-16 km/h
    ("grave",    [32, 36, 40],          52),   # ~exceso 13-20 km/h
    ("extrema",  [36, 40, V_MAX, V_MAX], 70),  # exceso >= 16 km/h
]

# Universo discretizado (0.5 km/h) para interpolar pertenencias
_U = np.arange(0, V_MAX + 0.5, 0.5)


def _mf(params: list[float]) -> np.ndarray:
    """Construye la curva de pertenencia (trap si 4 params, tri si 3)."""
    return fuzz.trapmf(_U, params) if len(params) == 4 else fuzz.trimf(_U, params)


def _pertenencia(params: list[float], v: float) -> float:
    return float(fuzz.interp_membership(_U, _mf(params), v))


# ----------------------------------------------------------------
#  Clasificación y sanción
# ----------------------------------------------------------------

def _grados_velocidad(v: float) -> dict:
    """Grados de pertenencia de las 3 categorías de velocidad en v."""
    return {
        "felicitacion": _pertenencia(_MF_FELIZ,  v),
        "normal":       _pertenencia(_MF_NORMAL, v),
        "multa":        _pertenencia(_MF_MULTA,  v),
    }


def _horas_sugeno(v: float) -> float:
    """
    Horas de sanción por inferencia Takagi–Sugeno de orden cero:
        horas = Σ μ_i(v)·c_i / Σ μ_i(v)
    con μ_i la pertenencia al sub-nivel i y c_i su consecuente singleton.
    Monótona y suave respecto a v (proporcional al exceso sobre 20 km/h).
    """
    mus = [_pertenencia(mf, v) for _, mf, _c in _SUBNIVELES]
    sings = [c for _, _mf, c in _SUBNIVELES]
    suma = sum(mus)
    if suma <= 1e-9:
        return 0.0
    return sum(m * c for m, c in zip(mus, sings)) / suma


def formatear_tiempo_sancion(horas_total: int) -> str:
    """Convierte horas a días y horas legibles. Ej: 38 → '1 día con 14 horas'."""
    if horas_total <= 0:
        return "Sin sanción"
    dias = horas_total // 24
    horas_rem = horas_total % 24
    if dias == 0:
        return f"{horas_rem} hora{'s' if horas_rem != 1 else ''}"
    elif horas_rem == 0:
        return f"{dias} día{'s' if dias != 1 else ''}"
    else:
        return (f"{dias} día{'s' if dias != 1 else ''} con "
                f"{horas_rem} hora{'s' if horas_rem != 1 else ''}")


def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna:
      - velocidad, clasificacion ('felicitacion'|'normal'|'multa')
      - horas_indisponibilidad (solo multa), tiempo_sancion (string)
      - grados_membresia (difusos), mensaje
    """
    v = float(np.clip(velocidad_kmh, 0, V_MAX))

    grados = _grados_velocidad(v)
    clasificacion = max(grados, key=grados.get)

    if clasificacion == "felicitacion":
        horas = 0
        mensaje = f"Felicitaciones — velocidad prudente ({v:.1f} km/h), sin sanción"
    elif clasificacion == "normal":
        horas = 0
        mensaje = f"Velocidad normal ({v:.1f} km/h) — sin sanción"
    else:  # multa
        horas = max(1, int(round(_horas_sugeno(v))))
        mensaje = (f"MULTA por exceso de velocidad ({v:.1f} km/h) — "
                   f"{formatear_tiempo_sancion(horas)} de indisponibilidad")

    return {
        "velocidad":              v,
        "clasificacion":          clasificacion,
        "horas_indisponibilidad": horas,
        "tiempo_sancion":         formatear_tiempo_sancion(horas),
        "grados_membresia":       {k: round(g, 3) for k, g in grados.items()},
        "mensaje":                mensaje,
    }


def _horas_politica(v: float) -> float:
    """Horas según la política completa: 0 si no es multa, Sugeno si lo es."""
    g = _grados_velocidad(v)
    if max(g, key=g.get) != "multa":
        return 0.0
    return _horas_sugeno(v)


# ──────────────────────────────────────────────────────────────────────────────
# Visualización del razonamiento difuso (para UI / correo)
# ──────────────────────────────────────────────────────────────────────────────
def _render_inferencia_png(velocidad_kmh: float) -> bytes:
    """
    Figura de dos paneles:
      Panel 1 — Entrada: pertenencias de velocidad (felicitacion/normal/multa),
                línea en la velocidad medida y los grados marcados.
      Panel 2 — Salida: curva de sanción horas(velocidad) (Takagi–Sugeno),
                monótona y proporcional, con el punto de operación marcado.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = float(np.clip(velocidad_kmh, 0, V_MAX))
    res = clasificar_velocidad(v)

    vu = np.arange(0, V_MAX + 0.5, 0.5)
    mf_feliz  = fuzz.trapmf(vu, _MF_FELIZ)
    mf_normal = fuzz.trimf(vu,  _MF_NORMAL)
    mf_multa  = fuzz.trapmf(vu, _MF_MULTA)

    C_FELIZ, C_NORMAL, C_MULTA = "#10b981", "#3b82f6", "#ef4444"
    C_OUT = "#8b5cf6"

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), dpi=110)
    fig.patch.set_facecolor("white")

    # ── Panel 1: VELOCIDAD (entrada) ──
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
    ax1.set_ylim(0, 1.18); ax1.set_xlim(0, V_MAX)
    ax1.legend(fontsize=8, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    ax1.grid(alpha=0.15)

    # ── Panel 2: SANCIÓN (salida Sugeno) — curva horas(velocidad) ──
    curva = np.array([_horas_politica(x) for x in vu])
    ax2.plot(vu, curva, color=C_OUT, lw=2.2, label="Sanción (Takagi–Sugeno)")
    ax2.fill_between(vu, curva, alpha=0.15, color=C_OUT)
    h_actual = res["horas_indisponibilidad"]
    ax2.axvline(v, color="#111827", ls="--", lw=1.2)
    if h_actual > 0:
        ax2.plot(v, h_actual, "o", color=C_OUT, ms=9, mec="white", mew=1.4, zorder=5)
        ax2.annotate(f"{h_actual} h\n({res['tiempo_sancion']})",
                     xy=(v, h_actual), xytext=(6, 6), textcoords="offset points",
                     fontsize=9, fontweight="bold", color=C_OUT)
    ax2.axvspan(0, 20, color="#10b981", alpha=0.05)
    ax2.text(10, H_MAX * 0.9, "zona permitida\n(≤20 km/h)", ha="center", va="top",
             fontsize=8, color="#059669")
    ax2.set_title("Salida — Horas de indisponibilidad (proporcional al exceso)",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlabel("velocidad (km/h)", fontsize=9)
    ax2.set_ylabel("horas", fontsize=9)
    ax2.set_ylim(0, H_MAX + 4); ax2.set_xlim(0, V_MAX)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
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
    """Razonamiento difuso como data-URI PNG base64 (para <img> en la UI)."""
    return "data:image/png;base64," + base64.b64encode(
        _render_inferencia_png(velocidad_kmh)).decode("ascii")


def guardar_inferencia_png(velocidad_kmh: float, ruta: str) -> str:
    """Razonamiento difuso guardado como archivo PNG (adjuntar en el correo)."""
    with open(ruta, "wb") as f:
        f.write(_render_inferencia_png(velocidad_kmh))
    return ruta


if __name__ == "__main__":
    casos = [5, 8, 12, 15, 18, 21, 24, 28, 32, 35, 38, 40]
    print("=" * 72)
    print("  LÓGICA DIFUSA — clasificación (Mamdani) + sanción proporcional (Sugeno)")
    print("=" * 72)
    for v in casos:
        r = clasificar_velocidad(v)
        print(f"\n  [{v:3d} km/h] → {r['clasificacion'].upper():12s} | {r['mensaje']}")
        print(f"    Membresía: felicitacion={r['grados_membresia']['felicitacion']:.2f}  "
              f"normal={r['grados_membresia']['normal']:.2f}  "
              f"multa={r['grados_membresia']['multa']:.2f}")
        if r["horas_indisponibilidad"] > 0:
            print(f"    Sanción: {r['tiempo_sancion']} ({r['horas_indisponibilidad']} h)")
