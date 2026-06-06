const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/events";

// Elements
const sourceTypeSel = document.getElementById('source-type');
const sourcePathGroup = document.getElementById('source-path-group');
const sourcePathInput = document.getElementById('source-path');
const sourceFileGroup = document.getElementById('source-file-group');
const sourceFileInput = document.getElementById('source-file');
const filePickerName = document.getElementById('file-picker-name');
const sourceImageGroup = document.getElementById('source-image-group');
const sourceImageInput = document.getElementById('source-image');
const imagePickerName = document.getElementById('image-picker-name');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const videoStream = document.getElementById('video-stream');
const videoPlaceholder = document.getElementById('video-placeholder');
const statusDot = document.getElementById('system-status-dot');
const statusText = document.getElementById('system-status-text');

// Dashboard Elements
const currentSpeedEl = document.getElementById('current-speed');
const currentPlateEl = document.getElementById('current-plate');
const currentClassificationEl = document.getElementById('current-classification');
const plateCard = document.getElementById('plate-card');
const eventsListEl = document.getElementById('events-list');

// Canvas Elements
const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');

// Playback Controls
const playbackControls = document.getElementById('playback-controls');
const btnRestart = document.getElementById('btn-restart');
const btnSeekBwd = document.getElementById('btn-seek-bwd');
const btnTogglePause = document.getElementById('btn-toggle-pause');
const btnSeekFwd = document.getElementById('btn-seek-fwd');

let ws = null;
let lines = {
    linea_a: { x1: 0, y1: 0, x2: 0, y2: 0 },
    linea_b: { x1: 0, y1: 0, x2: 0, y2: 0 }
};

// Canvas Interaction State
let draggingPoint = null; // { line: 'linea_a', point: 'p1' o 'p2' }
const POINT_RADIUS = 12;

// Update UI based on source type
sourceTypeSel.addEventListener('change', (e) => {
    sourcePathGroup.style.display = 'none';
    sourceFileGroup.style.display = 'none';
    sourceImageGroup.style.display = 'none';
    
    if (e.target.value === 'camera') {
        // Nada
    } else if (e.target.value === 'video') {
        sourceFileGroup.style.display = 'flex';
    } else if (e.target.value === 'image') {
        sourceImageGroup.style.display = 'flex';
    } else {
        sourcePathGroup.style.display = 'flex';
        sourcePathInput.placeholder = 'http://ip:port/video';
    }
});

// Update file picker name label
sourceFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        filePickerName.textContent = file.name;
    } else {
        filePickerName.textContent = "Ningún archivo seleccionado";
    }
});

sourceImageInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        imagePickerName.textContent = `${files.length} imagen(es) seleccionada(s)`;
    } else {
        imagePickerName.textContent = "Ninguna imagen seleccionada";
    }
});

function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// State for image testing mode
window.imageOcrResults = [];
window.currentImageIndex = 0;

function renderCurrentImage() {
    if (window.imageOcrResults.length === 0) return;
    const res = window.imageOcrResults[window.currentImageIndex];
    
    document.getElementById('image-counter').textContent = `${window.currentImageIndex + 1}/${window.imageOcrResults.length}`;
    
    if (res.error) {
        showToast("Error en imagen: " + res.error);
        return;
    }
    
    videoStream.src = "data:image/jpeg;base64," + res.image_b64;
    videoStream.style.display = 'block';
    videoPlaceholder.style.display = 'none';
    
    currentPlateEl.textContent = res.placa || "---";
    currentClassificationEl.textContent = res.confianza ? `CONF: ${(res.confianza*100).toFixed(1)}%` : "N/A";
    currentClassificationEl.style.color = "#00e5ff";
    currentSpeedEl.textContent = "---";
    plateCard.className = "card waiting";
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

document.getElementById('btn-prev-image').addEventListener('click', () => {
    if (window.currentImageIndex > 0) {
        window.currentImageIndex--;
        renderCurrentImage();
    }
});
document.getElementById('btn-next-image').addEventListener('click', () => {
    if (window.currentImageIndex < window.imageOcrResults.length - 1) {
        window.currentImageIndex++;
        renderCurrentImage();
    }
});

// Start Pipeline
btnStart.addEventListener('click', async () => {
    let fuente = 0;
    
    if (sourceTypeSel.value === 'image') {
        const files = sourceImageInput.files;
        if (files.length === 0) {
            showToast("Por favor selecciona al menos una imagen.");
            return;
        }
        
        showToast("Analizando imágenes...");
        btnStart.disabled = true;
        
        try {
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append("files", files[i]);
            }
            
            const uploadRes = await fetch(`${API_URL}/api/test-ocr`, {
                method: "POST",
                body: formData
            });
            const data = await uploadRes.json();
            
            btnStart.disabled = false;
            
            if (!data.success) {
                showToast("Error en análisis: " + data.message);
                return;
            }
            
            window.imageOcrResults = data.resultados;
            window.currentImageIndex = 0;
            
            document.getElementById('image-nav-group').style.display = 'flex';
            document.getElementById('btn-seek-bwd').style.display = 'none';
            document.getElementById('btn-toggle-pause').style.display = 'none';
            document.getElementById('btn-seek-fwd').style.display = 'none';
            document.getElementById('playback-speed-group').style.display = 'none';
            playbackControls.style.display = 'flex';
            
            renderCurrentImage();
            return; // Terminar aquí, no inicia websocket ni pipeline de video
            
        } catch (error) {
            showToast("Error al subir imágenes.");
            btnStart.disabled = false;
            return;
        }
    }
    
    // Ocultar nav de imágenes si estábamos en ese modo
    document.getElementById('image-nav-group').style.display = 'none';
    document.getElementById('btn-seek-bwd').style.display = 'block';
    document.getElementById('btn-toggle-pause').style.display = 'block';
    document.getElementById('btn-seek-fwd').style.display = 'block';
    document.getElementById('playback-speed-group').style.display = 'block';
    
    if (sourceTypeSel.value === 'video') {
        const file = sourceFileInput.files[0];
        if (!file) {
            showToast("Por favor selecciona un archivo de video.");
            return;
        }
        
        showToast("Subiendo video al servidor...");
        btnStart.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append("file", file);
            
            const uploadRes = await fetch(`${API_URL}/api/upload`, {
                method: "POST",
                body: formData
            });
            const uploadData = await uploadRes.json();
            
            if (!uploadData.success) {
                showToast("Error al subir el video: " + uploadData.message);
                btnStart.disabled = false;
                return;
            }
            fuente = uploadData.path;
        } catch (uploadError) {
            showToast("Fallo la conexión al subir el video.");
            btnStart.disabled = false;
            return;
        }
    } else if (sourceTypeSel.value === 'stream') {
        fuente = sourcePathInput.value.trim();
        if (!fuente) {
            showToast("Por favor ingresa una URL válida.");
            return;
        }
    }

    try {
        const res = await fetch(`${API_URL}/api/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fuente })
        });
        const data = await res.json();
        
        if (data.success) {
            btnStart.disabled = true;
            btnStop.disabled = false;
            
            // Start MJPEG Feed
            videoStream.src = `${API_URL}/video_feed?t=${new Date().getTime()}`;
            videoStream.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            
            // Save to localStorage
            if (sourceTypeSel.value === 'video') {
                localStorage.setItem('lastFuente', fuente);
            }
            
            checkStatusAndSyncUI();
            connectWebSocket();
            fetchLines(); // Load initial lines
        } else {
            showToast(data.message);
            btnStart.disabled = false;
        }
    } catch (error) {
        showToast("Error conectando con el backend.");
        btnStart.disabled = false;
    }
});

// Stop Pipeline
btnStop.addEventListener('click', async () => {
    try {
        await fetch(`${API_URL}/api/stop`, { method: 'POST' });
        btnStart.disabled = false;
        btnStop.disabled = true;
        
        videoStream.src = "";
        videoStream.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
        playbackControls.style.display = 'none';
        
        if (ws) ws.close();
        
        statusDot.className = 'status-dot';
        statusText.textContent = "Sistema Detenido";
    } catch (error) {
        showToast("Error deteniendo el pipeline.");
    }
});

// Playback Controls
async function sendPlaybackCmd(action) {
    try {
        const res = await fetch(`${API_URL}/api/playback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        const data = await res.json();
        if (action === 'toggle' && data.success) {
            btnTogglePause.textContent = data.paused ? '▶️' : '⏸';
            showToast(data.paused ? "Pausado" : "Reanudado");
        }
    } catch (e) {
        showToast("Error enviando comando de reproducción.");
    }
}

btnTogglePause.addEventListener('click', () => sendPlaybackCmd('toggle'));
btnSeekFwd.addEventListener('click', () => sendPlaybackCmd('seek_fwd'));
btnSeekBwd.addEventListener('click', () => sendPlaybackCmd('seek_bwd'));
btnRestart.addEventListener('click', () => sendPlaybackCmd('restart'));

const speedSelect = document.getElementById('playback-speed');
if (speedSelect) {
    speedSelect.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_URL}/api/playback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'set_speed', speed: parseFloat(e.target.value) })
            });
            showToast(`Velocidad: ${e.target.value}x`);
        } catch (error) {
            showToast("Error cambiando velocidad.");
        }
    });
}

// Status Sync
async function checkStatusAndSyncUI() {
    try {
        const res = await fetch(`${API_URL}/api/status`);
        const status = await res.json();
        
        if (status.running) {
            btnStart.disabled = true;
            btnStop.disabled = false;
            videoStream.src = `${API_URL}/video_feed?t=${new Date().getTime()}`;
            videoStream.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            
            if (status.es_archivo) {
                playbackControls.style.display = 'flex';
                btnTogglePause.textContent = status.paused ? '▶️' : '⏸';
            } else {
                playbackControls.style.display = 'none';
            }
            
            // Si no estaba conectado a WS, conectar
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                connectWebSocket();
                fetchLines();
            }
        } else {
            // Restore last selected video from local storage
            const lastFuente = localStorage.getItem('lastFuente');
            if (lastFuente) {
                sourceTypeSel.value = 'video';
                sourceTypeSel.dispatchEvent(new Event('change'));
                filePickerName.textContent = "Último video listo (haz clic en Iniciar)";
            }
        }
    } catch (e) {
        console.error("No se pudo obtener el estado:", e);
    }
}

// Initial Sync
window.addEventListener('DOMContentLoaded', checkStatusAndSyncUI);

// WebSocket Connection
function connectWebSocket() {
    if (ws) ws.close();
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        statusDot.className = 'status-dot active';
        statusText.textContent = "Conectado al Radar";
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleEvent(data);
    };
    
    ws.onclose = () => {
        statusDot.className = 'status-dot';
        statusText.textContent = "Desconectado";
        btnStart.disabled = false;
        btnStop.disabled = true;
    };
}

function handleEvent(msg) {
    const { type, payload } = msg;

    // El servidor manda dos tipos de mensajes: type='event' (eventos del pipeline)
    // y type='status' (cambios de estado del sistema).
    // Dentro de los 'event', payload.type es el subtipo real.

    if (type === 'event') {
        const evType = payload.type;

        if (evType === 'velocidad') {
            currentSpeedEl.textContent = parseFloat(payload.velocidad).toFixed(1);
            currentPlateEl.textContent = "BUSCANDO...";
            currentClassificationEl.textContent = "ANALIZANDO";
            currentClassificationEl.style.color = '#f0a500';
            plateCard.className = 'card waiting';
        }
        else if (evType === 'placa') {
            currentPlateEl.textContent = payload.placa;
        }
        else if (evType === 'registro_guardado') {
            const clasif = (payload.clasificacion || 'normal').toLowerCase();
            currentPlateEl.textContent = payload.placa;
            currentSpeedEl.textContent = parseFloat(payload.velocidad).toFixed(1);
            currentClassificationEl.textContent = clasif.toUpperCase();

            if (clasif === 'multa' || clasif === 'temerario') {
                currentClassificationEl.style.color = '#ff4d4d';
                plateCard.className = 'card danger';
                showToast(`⚠️ INFRACCION: ${payload.placa} a ${parseFloat(payload.velocidad).toFixed(1)} km/h`);
            } else if (clasif === 'advertencia') {
                currentClassificationEl.style.color = '#f0a500';
                plateCard.className = 'card warning';
                showToast(`⚠️ Advertencia: ${payload.placa} a ${parseFloat(payload.velocidad).toFixed(1)} km/h`);
            } else {
                currentClassificationEl.style.color = '#00e676';
                plateCard.className = 'card success';
            }

            addEventToList(payload);
        }
    }
    else if (type === 'status') {
        if (payload.state === 'ready') {
            currentSpeedEl.textContent = "0.0";
            currentPlateEl.textContent = "---";
            currentClassificationEl.textContent = "ESPERANDO";
            currentClassificationEl.style.color = '';
            plateCard.className = 'card waiting';
        } else if (payload.state === 'ended') {
            btnStop.click();
            showToast("Video finalizado.");
        }
    }
}

function addEventToList(payload) {
    const clasif = (payload.clasificacion || 'normal').toLowerCase();
    const vel = parseFloat(payload.velocidad).toFixed(1);
    const placa = payload.placa || '---';
    const hora = new Date().toLocaleTimeString('es-EC', {hour:'2-digit', minute:'2-digit', second:'2-digit'});

    let badgeColor = '#00e676';
    let borderColor = 'rgba(0,230,118,0.25)';
    if (clasif === 'multa' || clasif === 'temerario') {
        badgeColor = '#ff4d4d'; borderColor = 'rgba(255,77,77,0.25)';
    } else if (clasif === 'advertencia') {
        badgeColor = '#f0a500'; borderColor = 'rgba(240,165,0,0.25)';
    }

    const el = document.createElement('div');
    el.style.cssText = `display:flex; justify-content:space-between; align-items:center; 
        padding: 10px 14px; margin-bottom: 8px; border-radius: 8px;
        background: rgba(255,255,255,0.05); border-left: 3px solid ${borderColor};
        animation: slideIn 0.3s ease;`;
    el.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:2px">
            <span style="font-size:16px; font-weight:700; font-family:monospace; letter-spacing:2px; color:#fff">${placa}</span>
            <span style="font-size:11px; color:#aaa">${hora}</span>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:2px">
            <span style="font-size:12px; font-weight:700; color:${badgeColor}; text-transform:uppercase">${clasif}</span>
            <span style="font-size:13px; color:#ddd">${vel} <span style="font-size:10px;color:#aaa">km/h</span></span>
        </div>
    `;

    eventsListEl.prepend(el);
    // Mantener maximo 10 eventos
    while (eventsListEl.children.length > 10) {
        eventsListEl.lastChild.remove();
    }
}

// ----------------------------------------------------
// Canvas Interaction (Lines)
// ----------------------------------------------------

async function fetchLines() {
    try {
        const res = await fetch(`${API_URL}/api/lines`);
        const data = await res.json();
        lines = data;
        resizeCanvas();
    } catch (e) {
        console.error("Failed to fetch lines");
    }
}

async function saveLines() {
    try {
        await fetch(`${API_URL}/api/lines`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(lines)
        });
    } catch (e) {
        console.error("Failed to save lines");
    }
}

function resizeCanvas() {
    // The video stream might have a different intrinsic size vs display size
    // For simplicity, we make the canvas match the container's physical size
    // and scale coordinates accordingly, OR we assume coordinates are relative to intrinsic size.
    // In our backend, lines are based on the actual frame size (e.g. 640x480).
    // Let's make the canvas match the intrinsic video size so 1:1 mapping works.
    
    if (videoStream.style.display !== 'none') {
        const vw = videoStream.naturalWidth || 640;
        const vh = videoStream.naturalHeight || 480;
        
        if (vw > 0 && vh > 0) {
            canvas.width = vw;
            canvas.height = vh;
            
            // Adjust canvas CSS to fit exactly over the object-fit:contain image
            const rect = videoStream.getBoundingClientRect();
            const ratio = Math.min(rect.width / vw, rect.height / vh);
            const w = vw * ratio;
            const h = vh * ratio;
            
            canvas.style.width = `${w}px`;
            canvas.style.height = `${h}px`;
            // Center the canvas if needed based on object-fit
            canvas.style.left = `${(rect.width - w) / 2}px`;
            canvas.style.top = `${(rect.height - h) / 2}px`;
        }
    }
    drawLines();
}

videoStream.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);

function drawLines() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const drawLine = (line, color, isDrag) => {
        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.stroke();
        
        // Draw endpoints
        [ {x: line.x1, y: line.y1}, {x: line.x2, y: line.y2} ].forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, POINT_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });
    };
    
    // Highlight dragging point if any
    const colorA = draggingPoint?.line === 'linea_a' ? '#00b4ff' : '#ff6400';
    const colorB = draggingPoint?.line === 'linea_b' ? '#00b4ff' : '#ff0000';
    
    drawLine(lines.linea_a, colorA);
    drawLine(lines.linea_b, colorB);
}

// Mouse events on canvas
function getMousePos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

canvas.addEventListener('mousedown', (e) => {
    const pos = getMousePos(e);
    
    // Check which point is clicked
    for (const [lineKey, line] of Object.entries(lines)) {
        if (Math.hypot(pos.x - line.x1, pos.y - line.y1) < POINT_RADIUS * 2) {
            draggingPoint = { line: lineKey, point: 'p1' };
            drawLines();
            return;
        }
        if (Math.hypot(pos.x - line.x2, pos.y - line.y2) < POINT_RADIUS * 2) {
            draggingPoint = { line: lineKey, point: 'p2' };
            drawLines();
            return;
        }
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (!draggingPoint) return;
    
    const pos = getMousePos(e);
    const line = lines[draggingPoint.line];
    
    // Clamp to canvas bounds
    const cx = Math.max(0, Math.min(pos.x, canvas.width));
    const cy = Math.max(0, Math.min(pos.y, canvas.height));
    
    if (draggingPoint.point === 'p1') {
        line.x1 = cx;
        line.y1 = cy;
    } else {
        line.x2 = cx;
        line.y2 = cy;
    }
    drawLines();
});

canvas.addEventListener('mouseup', () => {
    if (draggingPoint) {
        draggingPoint = null;
        drawLines();
        saveLines(); // Notify backend of new positions
    }
});
canvas.addEventListener('mouseleave', () => {
    if (draggingPoint) {
        draggingPoint = null;
        drawLines();
        saveLines();
    }
});

// Interval to keep checking intrinsic size when playing stream
setInterval(() => {
    if (videoStream.style.display !== 'none' && videoStream.naturalWidth > 0) {
        if (canvas.width !== videoStream.naturalWidth || canvas.height !== videoStream.naturalHeight) {
            resizeCanvas();
        }
    }
}, 1000);
