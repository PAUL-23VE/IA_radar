"""
velocidad/logica_difusa.py
Sistema de inferencia difusa para clasificación de velocidad y cálculo de sanción.

Política del proyecto (zona controlada universitaria, límite 20 km/h):
  0  – 10 km/h  → FELICITACIÓN  (velocidad prudente, sin sanción)
  10 – 20 km/h  → NORMAL        (velocidad aceptable, sin sanción)
  > 20 km/h     → MULTA         (horas de indisponibilidad proporcionales al exceso)

══════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DEL SISTEMA DIFUSO (dos etapas)
══════════════════════════════════════════════════════════════════════════════

Etapa 1 — CLASIFICACIÓN (Mamdani, máxima pertenencia):
  Tres conjuntos difusos de velocidad de entrada con sus funciones de membresía:
    • Felicitación : trapezoidal [0, 0, 8, 12] km/h
    • Normal       : triangular  [9, 15, 21]   km/h
    • Multa        : trapezoidal [20, 25, 40, 40] km/h  ← rampa limpia, sin serrucho

  La categoría se elige por máxima pertenencia (defuzzificación por centroide).

Etapa 2 — SANCIÓN (Takagi–Sugeno de orden 0, solo si categoría = multa):
  Reglas IF-THEN explícitas sobre sub-niveles de multa con consecuentes singleton:

    REGLA 1: IF velocidad IS multa_leve     THEN horas = 5  h
    REGLA 2: IF velocidad IS multa_moderada THEN horas = 18 h
    REGLA 3: IF velocidad IS multa_alta     THEN horas = 34 h
    REGLA 4: IF velocidad IS multa_grave    THEN horas = 52 h
    REGLA 5: IF velocidad IS multa_extrema  THEN horas = 70 h

  Los sub-niveles teselan el rango 20–40 km/h con 50 % de solapamiento.
  La salida se calcula como promedio ponderado de los singletons:

        H(v) = Σ μᵢ(v) · cᵢ / Σ μᵢ(v)

  Esta combinación convexa de singletons estrictamente crecientes garantiza que
  H(v) sea ESTRICTAMENTE MONÓTONA (a mayor exceso sobre 20 km/h, mayor sanción),
  evitando las mesetas y micro-descensos no monótonos del centroide Mamdani
  (efecto «serrucho»), que violan el principio de proporcionalidad legal.
"""

import io
import base64

import numpy as np
import skfuzzy as fuzz


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del universo
# ─────────────────────────────────────────────────────────────────────────────
V_MAX = 40          # velocidad máxima del universo (km/h)
H_MAX = 72          # sanción máxima (horas = 3 días)

# Paso de discretización del universo continuo
_U = np.arange(0, V_MAX + 0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1 — Funciones de membresía de VELOCIDAD (3 categorías de entrada)
# ─────────────────────────────────────────────────────────────────────────────
#  • Felicitación : cubre 0–12 km/h  (meseta plena en 0–8, rampa descendente 8–12)
#  • Normal       : cubre 9–21 km/h  (triángulo centrado en 15)
#  • Multa        : cubre 20–40 km/h (rampa ascendente 20–25, meseta plena 25–40)
#    → La multa es una RAMPA LIMPIA (trapezoidal): sin serrucho, sin multimodalidad.
_MF_FELICITACION = [0,  0,  8, 12]         # trapezoidal
_MF_NORMAL       = [9, 15, 21]             # triangular
_MF_MULTA        = [20, 25, V_MAX, V_MAX]  # trapezoidal (rampa monótona)


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2 — Reglas Takagi–Sugeno (sub-niveles internos de multa)
# ─────────────────────────────────────────────────────────────────────────────
#  Cada regla define: (nombre, parámetros MF, consecuente singleton en horas)
#  Los triángulos se solapan un 50 % para una transición suave entre sub-niveles.
#
#  REGLA 1: IF v IS multa_leve     (20–28 km/h) THEN H = 5  h
#  REGLA 2: IF v IS multa_moderada (24–32 km/h) THEN H = 18 h
#  REGLA 3: IF v IS multa_alta     (28–36 km/h) THEN H = 34 h
#  REGLA 4: IF v IS multa_grave    (32–40 km/h) THEN H = 52 h
#  REGLA 5: IF v IS multa_extrema  (≥36 km/h)  THEN H = 70 h
_REGLAS_SUGENO = [
    # (nombre,        [params MF],            singleton_h)
    ("multa_leve",     [20, 24, 28],            5),   # exceso  1– 8 km/h → 5 h
    ("multa_moderada", [24, 28, 32],           18),   # exceso  5–12 km/h → 18 h
    ("multa_alta",     [28, 32, 36],           34),   # exceso  9–16 km/h → 34 h
    ("multa_grave",    [32, 36, 40],           52),   # exceso 13–20 km/h → 52 h
    ("multa_extrema",  [36, 40, V_MAX, V_MAX], 70),   # exceso ≥16 km/h  → 70 h
]


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _mf_array(params: list) -> np.ndarray:
    """Construye la curva de membresía sobre _U (trap si 4 params, tri si 3)."""
    return fuzz.trapmf(_U, params) if len(params) == 4 else fuzz.trimf(_U, params)


def _grado(params: list, v: float) -> float:
    """Grado de membresía de v en la función definida por params."""
    return float(fuzz.interp_membership(_U, _mf_array(params), v))


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1 — Clasificación (Mamdani / máxima pertenencia)
# ─────────────────────────────────────────────────────────────────────────────

def _grados_velocidad(v: float) -> dict:
    """
    Grados de membresía de las 3 categorías de velocidad en v.
    La categoría se asigna por máxima pertenencia (Mamdani).
    """
    return {
        "felicitacion": _grado(_MF_FELICITACION, v),
        "normal":       _grado(_MF_NORMAL,       v),
        "multa":        _grado(_MF_MULTA,         v),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2 — Sanción proporcional (Takagi–Sugeno, orden 0)
# ─────────────────────────────────────────────────────────────────────────────

def _horas_sugeno(v: float) -> float:
    """
    Evalúa las 5 reglas Sugeno y devuelve el promedio ponderado de singletons.

        H(v) = Σ μᵢ(v) · cᵢ  /  Σ μᵢ(v)

    La combinación convexa de singletons crecientes (c₁ < c₂ < … < c₅)
    garantiza que H(v) sea estrictamente monótona en el rango 20–40 km/h.
    """
    mus  = [_grado(mf, v) for _, mf, _ in _REGLAS_SUGENO]
    cs   = [c              for _, _,  c in _REGLAS_SUGENO]
    suma = sum(mus)
    if suma <= 1e-9:
        return 0.0
    return sum(m * c for m, c in zip(mus, cs)) / suma


def _horas_politica(v: float) -> float:
    """Horas según política completa: 0 si no es multa, Sugeno si lo es."""
    g = _grados_velocidad(v)
    if max(g, key=g.get) != "multa":
        return 0.0
    return _horas_sugeno(v)


# ─────────────────────────────────────────────────────────────────────────────
# Formateo de la sanción
# ─────────────────────────────────────────────────────────────────────────────

def formatear_tiempo_sancion(horas_total: int) -> str:
    """
    Convierte horas a formato legible en días y horas.
    Ejemplos: 38 → '1 día con 14 horas' | 24 → '1 día' | 6 → '6 horas'
    """
    if horas_total <= 0:
        return "Sin sanción"
    dias     = horas_total // 24
    horas_r  = horas_total % 24
    if dias == 0:
        return f"{horas_r} hora{'s' if horas_r != 1 else ''}"
    elif horas_r == 0:
        return f"{dias} día{'s' if dias != 1 else ''}"
    else:
        return (f"{dias} día{'s' if dias != 1 else ''} con "
                f"{horas_r} hora{'s' if horas_r != 1 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# Interfaz pública principal
# ─────────────────────────────────────────────────────────────────────────────

def clasificar_velocidad(velocidad_kmh: float) -> dict:
    """
    Recibe la velocidad en km/h y retorna un diccionario con:
      - velocidad              : valor clampado a [0, V_MAX]
      - clasificacion          : 'felicitacion' | 'normal' | 'multa'
      - horas_indisponibilidad : horas de sanción (0 si no es multa)
      - tiempo_sancion         : cadena formateada ('X días con Y horas')
      - grados_membresia       : dict con grados difusos de cada categoría
      - mensaje                : descripción para mostrar al usuario
    """
    v = float(np.clip(velocidad_kmh, 0, V_MAX))

    grados       = _grados_velocidad(v)
    clasificacion = max(grados, key=grados.get)

    if clasificacion == "felicitacion":
        horas   = 0
        mensaje = f"Felicitaciones — velocidad prudente ({v:.1f} km/h), sin sanción"
    elif clasificacion == "normal":
        horas   = 0
        mensaje = f"Velocidad normal ({v:.1f} km/h) — sin sanción"
    else:  # multa
        horas   = max(1, int(round(_horas_sugeno(v))))
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


# ─────────────────────────────────────────────────────────────────────────────
# Visualización del razonamiento difuso (para UI / correo)
# ─────────────────────────────────────────────────────────────────────────────

def _render_inferencia_png(velocidad_kmh: float) -> bytes:
    """
    Genera la figura de razonamiento difuso en dos paneles:

    Panel 1 — Entrada (velocidad):
        Tres funciones de membresía limpias (felicitación / normal / multa),
        línea vertical en la velocidad medida y puntos de activación.
        La multa es una RAMPA TRAPEZOIDAL limpia (sin serrucho).

    Panel 2 — Salida (sanción Takagi–Sugeno):
        Curva H(v) = horas de indisponibilidad en función de la velocidad.
        La curva es estrictamente monótona y proporcional al exceso sobre 20 km/h.
        Se marcan la velocidad medida y las horas de sanción resultantes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v   = float(np.clip(velocidad_kmh, 0, V_MAX))
    res = clasificar_velocidad(v)

    # Arrays de membresía sobre el universo
    vu         = np.arange(0, V_MAX + 0.5, 0.5)
    mf_feliz   = fuzz.trapmf(vu, _MF_FELICITACION)
    mf_normal  = fuzz.trimf(vu,  _MF_NORMAL)
    mf_multa   = fuzz.trapmf(vu, _MF_MULTA)

    # Curva de sanción Sugeno (para todos los puntos del universo)
    curva_h = np.array([_horas_politica(x) for x in vu])

    C_FELIZ  = "#10b981"   # verde esmeralda
    C_NORMAL = "#3b82f6"   # azul
    C_MULTA  = "#ef4444"   # rojo
    C_OUT    = "#8b5cf6"   # violeta

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.8), dpi=110)
    fig.patch.set_facecolor("white")

    # ── Panel 1: funciones de membresía de VELOCIDAD (entrada) ──────────────
    ax1.plot(vu, mf_feliz,  color=C_FELIZ,  lw=2.0, label="Felicitación (0–10)")
    ax1.fill_between(vu, mf_feliz,  alpha=0.13, color=C_FELIZ)
    ax1.plot(vu, mf_normal, color=C_NORMAL, lw=2.0, label="Normal (10–20)")
    ax1.fill_between(vu, mf_normal, alpha=0.13, color=C_NORMAL)
    ax1.plot(vu, mf_multa,  color=C_MULTA,  lw=2.0, label="Multa (>20)")
    ax1.fill_between(vu, mf_multa,  alpha=0.13, color=C_MULTA)

    # Línea vertical en la velocidad medida
    ax1.axvline(v, color="#111827", ls="--", lw=1.5)
    ax1.text(v, 1.07, f"{v:.1f} km/h", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color="#111827")

    # Marcar grados de activación de cada categoría
    for cat, mf_arr, col in (
        ("felicitacion", mf_feliz,  C_FELIZ),
        ("normal",       mf_normal, C_NORMAL),
        ("multa",        mf_multa,  C_MULTA),
    ):
        g = res["grados_membresia"][cat]
        if g > 0.01:
            ax1.plot(v, g, "o", color=col, ms=8, mec="white", mew=1.5, zorder=5)

    ax1.set_title("Entrada difusa — Velocidad", fontsize=11, fontweight="bold", loc="left")
    ax1.set_xlabel("km/h", fontsize=9)
    ax1.set_ylabel("Pertenencia", fontsize=9)
    ax1.set_ylim(0, 1.22)
    ax1.set_xlim(0, V_MAX)
    ax1.legend(fontsize=8.5, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    ax1.grid(alpha=0.15)

    # ── Panel 2: curva de sanción Takagi–Sugeno (salida) ─────────────────────
    ax2.plot(vu, curva_h, color=C_OUT, lw=2.4, label="Sanción (Takagi–Sugeno)")
    ax2.fill_between(vu, curva_h, alpha=0.15, color=C_OUT)

    h_actual = res["horas_indisponibilidad"]

    # Zona permitida (≤ 20 km/h)
    ax2.axvspan(0, 20, color=C_FELIZ, alpha=0.07)
    ax2.text(10, H_MAX * 0.88, "zona\npermitida\n(≤ 20 km/h)",
             ha="center", va="top", fontsize=8, color="#059669")

    # Línea vertical y punto de operación
    ax2.axvline(v, color="#111827", ls="--", lw=1.3)
    if h_actual > 0:
        ax2.plot(v, h_actual, "o", color=C_OUT, ms=10,
                 mec="white", mew=1.5, zorder=5)
        ax2.annotate(
            f"{h_actual} h\n({res['tiempo_sancion']})",
            xy=(v, h_actual),
            xytext=(8, 6), textcoords="offset points",
            fontsize=9, fontweight="bold", color=C_OUT,
        )

    ax2.set_title(
        "Salida — Horas de indisponibilidad (proporcional al exceso)",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax2.set_xlabel("velocidad (km/h)", fontsize=9)
    ax2.set_ylabel("horas", fontsize=9)
    ax2.set_ylim(0, H_MAX + 5)
    ax2.set_xlim(0, V_MAX)
    ax2.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax2.grid(alpha=0.15)

    # Título general con resultado
    clasif  = res["clasificacion"].upper()
    col_t   = {"FELICITACION": C_FELIZ, "NORMAL": C_NORMAL, "MULTA": C_MULTA}.get(clasif, "#111")
    fig.suptitle(
        f"Resultado: {clasif}  ·  {res['tiempo_sancion']}",
        fontsize=12, fontweight="bold", color=col_t,
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# Prueba por línea de comandos
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    casos = [5, 8, 12, 15, 18, 21, 24, 28, 32, 35, 38, 40]
    print("=" * 72)
    print("  SISTEMA DIFUSO — Mamdani (clasificación) + Sugeno (sanción proporcional)")
    print("=" * 72)
    print(f"\n  {'km/h':>5}  {'Categoría':<14} {'Horas':>6}  Sanción")
    print(f"  {'-'*5}  {'-'*14} {'-'*6}  {'-'*25}")
    for v in casos:
        r = clasificar_velocidad(v)
        g = r["grados_membresia"]
        print(f"  {v:>5}  {r['clasificacion'].upper():<14} {r['horas_indisponibilidad']:>6}  "
              f"{r['tiempo_sancion']}")
        print(f"         μ_feliz={g['felicitacion']:.2f}  μ_normal={g['normal']:.2f}  "
              f"μ_multa={g['multa']:.2f}")

    # Verificar monotonicidad
    horas = [clasificar_velocidad(v)["horas_indisponibilidad"] for v in range(20, 41)]
    ok = all(horas[i] <= horas[i+1] for i in range(len(horas) - 1))
    print(f"\n  Monotonicidad Sugeno (20→40 km/h): {'✅ OK' if ok else '❌ FALLO'}")
    print(f"  Curva: {horas}")
