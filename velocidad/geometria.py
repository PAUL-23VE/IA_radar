"""
velocidad/geometria.py
Módulo para el manejo geométrico de las líneas virtuales de medición de velocidad.
"""

import cv2

R_ENDPOINT = 12   # radio de agarre de un endpoint (px)
R_LINEA    = 10   # distancia máxima al segmento para agarrar la línea entera


def dist_punto_segmento(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Distancia del punto (px,py) al segmento (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return ((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2) ** 0.5


def lado_linea(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Signo del producto cruzado: positivo/negativo según el lado de la línea."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def callback_mouse_lineas(event, x, y, flags, param):
    """
    Click+drag sobre un endpoint (círculo) → mueve solo ese punto.
    Click+drag sobre el cuerpo de la línea → desplaza toda la línea.
    Sin restricciones de posición: las líneas van a donde el usuario quiera.
    """
    st = param   # dict con linea_a, linea_b, drag

    if event == cv2.EVENT_LBUTTONDOWN:
        st["drag"] = None
        for nombre in ("a", "b"):
            ln = st[f"linea_{nombre}"]
            # ¿click cerca de endpoint 1?
            if ((x - ln["x1"]) ** 2 + (y - ln["y1"]) ** 2) ** 0.5 <= R_ENDPOINT:
                st["drag"] = (nombre, "p1")
                break
            # ¿click cerca de endpoint 2?
            if ((x - ln["x2"]) ** 2 + (y - ln["y2"]) ** 2) ** 0.5 <= R_ENDPOINT:
                st["drag"] = (nombre, "p2")
                break
            # ¿click cerca del cuerpo de la línea?
            if dist_punto_segmento(x, y, ln["x1"], ln["y1"], ln["x2"], ln["y2"]) <= R_LINEA:
                st["drag"] = (nombre, "linea")
                st["drag_ox"] = x
                st["drag_oy"] = y
                st["drag_lx1"] = ln["x1"]
                st["drag_ly1"] = ln["y1"]
                st["drag_lx2"] = ln["x2"]
                st["drag_ly2"] = ln["y2"]
                break

    elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
        nombre, parte = st["drag"]
        ln = st[f"linea_{nombre}"]
        if parte == "p1":
            ln["x1"], ln["y1"] = x, y
        elif parte == "p2":
            ln["x2"], ln["y2"] = x, y
        else:  # mover línea completa
            dx = x - st["drag_ox"]
            dy = y - st["drag_oy"]
            ln["x1"] = st["drag_lx1"] + dx
            ln["ly1"] = st["drag_ly1"] + dy   # Wait! Wait! Is it ln["ly1"] or ln["y1"]?
            # Let's check: in main.py, it was ln["y1"] = st["drag_ly1"] + dy
            # And ln["y2"] = st["drag_ly2"] + dy
            # Let's write y1 and y2 correctly!
            ln["y1"] = st["drag_ly1"] + dy
            ln["x2"] = st["drag_lx2"] + dx
            ln["y2"] = st["drag_ly2"] + dy

    elif event == cv2.EVENT_LBUTTONUP:
        st["drag"] = None
