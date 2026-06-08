"""
api/email_service.py
Envio de correos por SMTP.

Se invoca desde pipeline._registrar_track() en un hilo separado por cada
vehiculo detectado (uno por auto), de modo que no bloquea el pipeline principal.
La configuración SMTP sale de config.py (variables .env).
"""

from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
import smtplib
import ssl

from config import settings


@dataclass(frozen=True)
class EmailPayload:
    to: str
    subject: str
    body: str                                    # texto plano (fallback)
    html: str | None = None                      # version HTML opcional
    inline_images: dict | None = None            # {cid: ruta_absoluta} para <img src="cid:...">


def send_email(payload: EmailPayload) -> None:
    import os
    if os.environ.get("RADAR_NO_EMAIL"):
        print(f"[MAIL] (deshabilitado RADAR_NO_EMAIL) habria enviado subject={payload.subject!r}")
        return
    print(f"[MAIL] enviando -> to={payload.to} subject={payload.subject!r} host={settings.SMTP_HOST}:{settings.SMTP_PORT}")

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = payload.to
    message["Subject"] = payload.subject
    message.set_content(payload.body)            # parte texto plano (fallback)

    if payload.html:
        message.add_alternative(payload.html, subtype="html")
        if payload.inline_images:
            html_part = message.get_payload()[-1]
            for cid, ruta in payload.inline_images.items():
                try:
                    data = Path(ruta).read_bytes()
                except OSError:
                    continue                     # imagen faltante -> se omite
                html_part.add_related(
                    data, maintype="image", subtype="jpeg", cid=f"<{cid}>")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(message)
        print(f"[MAIL] OK -> {payload.to}")
    except Exception as e:
        print(f"[MAIL] Error enviando correo: {e}")


# ── Firma del Grupo D ──────────────────────────────────────────────────────────
_FIRMA_HTML = """
<div style="margin-top:24px; padding-top:16px; border-top: 1px solid #e0e0e0;
            font-family: 'Segoe UI', sans-serif; font-size:12px; color:#999; line-height:1.7;">
  <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <div style="background: linear-gradient(135deg,#1a1a2e,#16213e); color:#fff; padding:8px 14px;
                border-radius:8px; font-weight:700; letter-spacing:2px; font-size:11px;">
      📡 RADAR AI · UTA
    </div>
    <div>
      <b style="color:#555;">Grupo D</b> — Sistema Inteligente de Monitoreo Vehicular<br>
      <span style="color:#aaa;">
        Manjarres Quintero, David Oswaldo ·
        Velastegui Eugenio, Anthony Paul
      </span>
    </div>
  </div>
  <div style="margin-top:8px; font-size:11px; color:#bbb;">
    Generado automáticamente · No responder a este correo
  </div>
</div>
"""


def _badge_html(clasificacion: str) -> str:
    """Retorna una etiqueta visual HTML según la clasificación."""
    colores = {
        "FELICITACION": ("#d4edda", "#155724", "✓ FELICITACIÓN"),
        "NORMAL":       ("#d1ecf1", "#0c5460", "● NORMAL"),
        "ADVERTENCIA":  ("#fff3cd", "#856404", "⚠ ADVERTENCIA"),
        "MULTA":        ("#f8d7da", "#721c24", "✗ MULTA / INFRACCIÓN"),
    }
    bg, fg, texto = colores.get(clasificacion, ("#e2e3e5", "#383d41", clasificacion))
    return (f'<span style="background:{bg}; color:{fg}; padding:4px 12px; border-radius:20px;'
            f' font-weight:700; font-size:13px; display:inline-block;">{texto}</span>')


def build_detection_html(data: dict):
    """Construye el HTML del correo y el mapa de imágenes inline."""
    placa = data.get("placa") or "—"
    placa_fmt = f"{placa[:3]}-{placa[3:]}" if len(placa) > 3 else placa
    velocidad = float(data.get("velocidad_kmh", 0))
    clasificacion = str(data.get("clasificacion", "normal")).upper()
    horas = int(data.get("horas", 0))
    tiempo_sancion = data.get("tiempo_sancion", "Sin sanción")

    # Colores de encabezado según gravedad
    color_map = {
        "FELICITACION": "#27ae60",
        "NORMAL":       "#2980b9",
        "ADVERTENCIA":  "#e67e22",
        "MULTA":        "#c0392b",
    }
    color_header = color_map.get(clasificacion, "#34495e")

    # Imagen de la captura como adjunto inline
    image_map = {}
    captura_html = ""
    ruta_cap = data.get("ruta_captura")
    if ruta_cap:
        cid = "captura_evento"
        image_map[cid] = ruta_cap
        captura_html = f"""
        <div style="text-align:center; margin:20px 0 4px;">
          <img src="cid:{cid}" alt="Captura del Evento" 
               style="max-width:100%; max-height:300px; border:2px solid #ddd;
                      border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
          <div style="font-size:11px; color:#aaa; margin-top:6px;">Captura del vehículo al cruzar la línea B</div>
        </div>"""

    # Gráfica del razonamiento difuso (membresías + defuzzificación) como adjunto inline
    grafica_html = ""
    ruta_graf = data.get("ruta_grafica")
    if ruta_graf:
        cid_g = "grafica_difuso"
        image_map[cid_g] = ruta_graf
        grafica_html = f"""
        <h3 style="font-size:14px; font-weight:700; color:#333; margin:24px 0 12px;
                   text-transform:uppercase; letter-spacing:1px; border-bottom:2px solid #eee; padding-bottom:8px;">
          Razonamiento de Lógica Difusa
        </h3>
        <div style="text-align:center; margin:8px 0;">
          <img src="cid:{cid_g}" alt="Inferencia difusa"
               style="max-width:100%; border:1px solid #eee; border-radius:10px;">
          <div style="font-size:11px; color:#aaa; margin-top:6px;">
            Membresías de velocidad y horas · centroide = sanción defuzzificada
          </div>
        </div>"""

    # Fila de sanción (solo si aplica)
    if horas > 0:
        sancion_fila = f"""
      <tr style="background:#fff8f8;">
        <td style="padding:10px 12px; color:#777; border-bottom:1px solid #f0f0f0; font-size:13px;">
          Sanción (Indisponibilidad)
        </td>
        <td style="padding:10px 12px; border-bottom:1px solid #f0f0f0;">
          <b style="color:#c0392b;">{tiempo_sancion}</b>
          <span style="font-size:11px; color:#aaa; margin-left:6px;">({horas} horas totales)</span>
        </td>
      </tr>"""
    else:
        sancion_fila = f"""
      <tr>
        <td style="padding:10px 12px; color:#777; border-bottom:1px solid #f0f0f0; font-size:13px;">
          Sanción
        </td>
        <td style="padding:10px 12px; border-bottom:1px solid #f0f0f0; color:#27ae60;">
          <b>Sin sanción</b>
        </td>
      </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<body style="margin:0; padding:0; background:#f5f5f5; font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
<div style="max-width:640px; margin:30px auto; background:#fff; border-radius:12px;
            box-shadow:0 4px 20px rgba(0,0,0,0.12); overflow:hidden; border:1px solid #e0e0e0;">

  <!-- Encabezado -->
  <div style="background:{color_header}; color:#fff; padding:24px 28px;">
    <div style="font-size:11px; letter-spacing:3px; opacity:0.8; margin-bottom:6px;">
      SISTEMA DE MONITOREO VEHICULAR · UTA · GRUPO D
    </div>
    <div style="font-size:34px; font-weight:800; letter-spacing:4px; font-family:monospace;">
      {placa_fmt}
    </div>
    <div style="margin-top:10px;">
      {_badge_html(clasificacion)}
    </div>
  </div>

  <!-- Cuerpo -->
  <div style="padding:24px 28px;">

    <!-- Imagen de captura -->
    {captura_html}

    <h3 style="font-size:14px; font-weight:700; color:#333; margin:20px 0 12px;
               text-transform:uppercase; letter-spacing:1px; border-bottom:2px solid #eee; padding-bottom:8px;">
      Detalles del Evento
    </h3>

    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <tr>
        <td style="padding:10px 12px; color:#777; border-bottom:1px solid #f0f0f0; width:45%;">Velocidad Registrada</td>
        <td style="padding:10px 12px; border-bottom:1px solid #f0f0f0;"><b>{velocidad:.1f} km/h</b></td>
      </tr>
      <tr>
        <td style="padding:10px 12px; color:#777; border-bottom:1px solid #f0f0f0;">Clasificación Difusa</td>
        <td style="padding:10px 12px; border-bottom:1px solid #f0f0f0;">{_badge_html(clasificacion)}</td>
      </tr>
      <tr>
        <td style="padding:10px 12px; color:#777; border-bottom:1px solid #f0f0f0;">Rango de Velocidad</td>
        <td style="padding:10px 12px; border-bottom:1px solid #f0f0f0; color:#555;">
          {"0 – 10 km/h (Felicitación · zona segura)" if velocidad <= 10 else
           "10 – 20 km/h (Velocidad normal)" if velocidad <= 20 else
           "> 20 km/h (Multa · exceso de velocidad)"}
        </td>
      </tr>
      {sancion_fila}
    </table>

    {grafica_html}

    {_firma_html_footer()}
  </div>
</div>
</body>
</html>"""
    return html, image_map


def _firma_html_footer() -> str:
    return _FIRMA_HTML


def enviar_notificacion_asincrona(data: dict):
    """
    Genera y envía el correo de notificación para un vehículo detectado.
    Se invoca en un hilo separado por cada evento (auto que cruza las líneas).
    """
    html, inline_images = build_detection_html(data)

    placa = data.get("placa") or "Desconocida"
    clasificacion = data.get("clasificacion", "normal").upper()
    velocidad = data.get("velocidad_kmh", 0)
    tiempo_sancion = data.get("tiempo_sancion", "")

    # Asunto claro con toda la info (incluye GRUPO D en el titulo)
    asunto = f"[RADAR-UTA · GRUPO D] {clasificacion} · {placa} a {float(velocidad):.1f} km/h"
    if tiempo_sancion and tiempo_sancion != "Sin sanción":
        asunto += f" · Sanción: {tiempo_sancion}"

    cuerpo_texto = (
        f"Vehículo detectado: {placa}\n"
        f"Velocidad: {float(velocidad):.1f} km/h\n"
        f"Clasificación: {clasificacion}\n"
        f"Sanción: {tiempo_sancion or 'Sin sanción'}\n"
        f"\nGenerado por el Sistema de Radar AI - UTA Grupo D"
    )

    payload = EmailPayload(
        to=settings.ENVIO_INFRACCIONES_A,
        subject=asunto,
        body=cuerpo_texto,
        html=html,
        inline_images=inline_images if inline_images else None
    )
    send_email(payload)
