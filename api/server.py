import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
import shutil

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

from api.pipeline import pipeline

app = FastAPI(title="Radar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # We need to make sure we don't block
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

# Callback para puente entre pipeline (thread síncrono) y websocket (asíncrono)
def on_pipeline_event(event_type, payload):
    # Esto corre en el thread del pipeline, por lo que usamos asyncio para mandar al loop principal
    msg = {"type": event_type, "payload": payload}
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast(msg))
    except RuntimeError:
        # Si no hay loop corriendo en este thread, buscamos el principal
        pass

pipeline.on_event(on_pipeline_event)


async def video_stream_generator():
    while True:
        frame_bytes = pipeline.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.033)  # cap ~30fps
        else:
            await asyncio.sleep(0.01)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"success": True, "path": os.path.abspath(file_path)}
    except Exception as e:
        return {"success": False, "message": str(e)}


class StartRequest(BaseModel):
    fuente: str | int

@app.post("/api/start")
async def start_pipeline(req: StartRequest):
    # Si la fuente es un número string ("0"), lo parseamos
    fuente = req.fuente
    if isinstance(fuente, str) and fuente.isdigit():
        fuente = int(fuente)
        
    success = pipeline.start(fuente=fuente)
    return {"success": success, "message": "Iniciado" if success else "No se pudo iniciar o ya estaba corriendo"}

@app.post("/api/stop")
async def stop_pipeline():
    pipeline.stop()
    return {"success": True}

class PlaybackRequest(BaseModel):
    action: str  # 'pause', 'resume', 'toggle', 'seek_fwd', 'seek_bwd', 'restart'

@app.post("/api/playback")
async def playback_control(req: PlaybackRequest):
    if req.action == "toggle":
        is_paused = pipeline.toggle_pause()
        return {"success": True, "paused": is_paused}
    elif req.action == "pause":
        pipeline.paused = True
        return {"success": True, "paused": True}
    elif req.action == "resume":
        pipeline.paused = False
        return {"success": True, "paused": False}
    elif req.action == "seek_fwd":
        success = pipeline.seek(5.0)
        return {"success": success}
    elif req.action == "seek_bwd":
        success = pipeline.seek(-5.0)
        return {"success": success}
    elif req.action == "restart":
        success = pipeline.restart_video()
        return {"success": success}
    return {"success": False, "message": "Acción desconocida"}

@app.get("/api/status")
async def get_status():
    return pipeline.get_status()

class Point(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class LinesRequest(BaseModel):
    linea_a: Point
    linea_b: Point

@app.get("/api/lines")
async def get_lines():
    return pipeline.get_lines()

@app.post("/api/lines")
async def set_lines(req: LinesRequest):
    pipeline.set_lines(req.linea_a.model_dump(), req.linea_b.model_dump())
    return {"success": True}

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Mantenemos viva la conexión y podemos recibir comandos si queremos
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
