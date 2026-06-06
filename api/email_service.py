"""
api/email_service.py
Envio de correos por SMTP.
"""

from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
import smtplib
import ssl

# Configuración quemada para facilitar las pruebas
SMTP_HOST = "smtp.office365.com"
SMTP_PORT = 587
SMTP_USER = "dmanjarres6065@uta.edu.ec"
SMTP_PASSWORD = "hqgqxvpwpzwjyxyl"
SMTP_FROM = "dmanjarres6065@uta.edu.ec"
ENVIO_INFRACCIONES_A = "davidmanjarres2004@gmail.com"

@dataclass(frozen=True)
class EmailPayload:
    to: str
    subject: str
    body: str                                    # texto plano (fallback)
    html: str | None = None                      # version HTML opcional
    inline_images: dict | None = None            # {cid: ruta_absoluta} para <img src="cid:...">

def send_email(payload: EmailPayload) -> None:
    print(f"[MAIL] enviando -> to={payload.to} subject={payload.subject!r} host={SMTP_HOST}:{SMTP_PORT}")

    message = EmailMessage()
    message["From"] = SMTP_FROM
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
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(message)
        print(f"[MAIL] OK -> {payload.to}")
    except Exception as e:
        print(f"[MAIL] Error enviando correo: {e}")

_INTEGRANTES_HTML = (
    '<div style="font-size:12px;opacity:.9;line-height:1.6;margin-top:10px;">'
    'SISTEMA DE MONITOREO VEHICULAR · UTA<br>'
    'Grupo D<br>Integrantes:<br>Manjarres Quintero David Oswaldo<br>Velastegui Eugenio Anthony Paul</div>'
)

def _header_html(color: str, placa_fmt: str, subtitulo: str) -> str:
    return (
        f'<div style="background:{color};color:#fff;padding:16px 20px;border-radius:8px 8px 0 0; border: 2px solid #222; box-shadow: inset 0px -3px 0px rgba(0,0,0,0.2);">'
        f'<div style="font-size:26px;font-weight:bold;letter-spacing:2px; font-family: monospace;">{placa_fmt}</div>'
        f'<div style="font-size:13px;margin-top:2px; font-style: italic;">{subtitulo}</div>'
        f'{_INTEGRANTES_HTML}</div>'
    )

def build_detection_html(data: dict):
    placa = data.get("placa") or "—"
    placa_fmt = f"{placa[:3]}-{placa[3:]}" if len(placa) > 3 else placa
    velocidad = float(data.get("velocidad_kmh", 0))
    clasificacion = str(data.get("clasificacion", "normal")).upper()
    horas = int(data.get("horas", 0))
    
    color = "#c0392b" if clasificacion == "MULTA" else "#27ae60"
    
    # image_map para la captura del evento
    image_map = {}
    captura_html = ""
    ruta_cap = data.get("ruta_captura")
    if ruta_cap:
        cid = "captura01"
        image_map[cid] = ruta_cap
        captura_html = f'<div style="text-align: center; margin-top: 15px;"><img src="cid:{cid}" alt="Captura Evento" style="max-width:100%; border:2px solid #555; border-radius:6px; display:block;"></div>'

    html = f"""\
<div style="font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;max-width:680px;margin:auto;color:#222; background: #fafafa; border-radius: 8px; border: 1px solid #ddd;">
  {_header_html(color, placa_fmt, f"Alerta de Monitoreo · <b>ESTADO: {clasificacion}</b>")}
  <div style="padding:18px 20px;">
    
    <h3 style="margin:20px 0 8px;font-size:16px;color:#333; border-bottom: 2px solid #ccc; padding-bottom: 4px;">Detalles del Evento</h3>
    <table style="font-size:14px;line-height:1.8; width: 100%; border-collapse: collapse;">
      <tr style="border-bottom: 1px solid #eee;"><td style="color:#666; width: 50%;">Velocidad Registrada</td>
          <td><b>{velocidad:.1f} km/h</b></td></tr>
      <tr style="border-bottom: 1px solid #eee;"><td style="color:#666;">Clasificación (Lógica Difusa)</td>
          <td><b style="color:{color};">{clasificacion}</b></td></tr>
      <tr style="border-bottom: 1px solid #eee;"><td style="color:#666;">Sanción Sugerida</td>
          <td><b>{f"{horas} horas de indisponibilidad" if horas > 0 else "Sin sanción (Advertencia/Normal)"}</b></td></tr>
    </table>
    
    {captura_html}
    
    <div style="margin-top: 25px; font-size: 11px; color: #888; text-align: center;">
        Generado automáticamente por el Sistema Inteligente de Radar de Placas.
    </div>
  </div>
</div>"""
    return html, image_map

def enviar_notificacion_asincrona(data: dict):
    """Genera y envía el correo con la plantilla del Grupo D."""
    html, inline_images = build_detection_html(data)
    
    placa = data.get("placa") or "Desconocida"
    clasificacion = data.get("clasificacion", "NORMAL").upper()
    
    asunto = f"[RADAR-UTA] Reporte Vehículo {placa} - {clasificacion}"
    cuerpo_texto = f"El vehículo {placa} fue detectado a {data.get('velocidad_kmh')} km/h. Clasificación: {clasificacion}."
    
    payload = EmailPayload(
        to=ENVIO_INFRACCIONES_A,
        subject=asunto,
        body=cuerpo_texto,
        html=html,
        inline_images=inline_images
    )
    send_email(payload)
