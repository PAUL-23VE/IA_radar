const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/events";

// Elements
const sourceTypeSel = document.getElementById('source-type');
const sourcePathGroup = document.getElementById('source-path-group');
const sourcePathInput = document.getElementById('source-path');
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
    if (e.target.value === 'camera') {
        sourcePathGroup.style.display = 'none';
    } else {
        sourcePathGroup.style.display = 'flex';
        sourcePathInput.placeholder = e.target.value === 'video' ? '/ruta/al/video.mp4' : 'http://ip:port/video';
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

// Start Pipeline
btnStart.addEventListener('click', async () => {
    let fuente = 0;
    if (sourceTypeSel.value === 'video' || sourceTypeSel.value === 'stream') {
        fuente = sourcePathInput.value.trim();
        if (!fuente) {
            showToast("Por favor ingresa una ruta o URL válida.");
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
            
            connectWebSocket();
            fetchLines(); // Load initial lines
        } else {
            showToast(data.message);
        }
    } catch (error) {
        showToast("Error conectando con el backend.");
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
        
        if (ws) ws.close();
        
        statusDot.className = 'status-dot';
        statusText.textContent = "Sistema Detenido";
    } catch (error) {
        showToast("Error deteniendo el pipeline.");
    }
});

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
    
    if (type === 'velocidad') {
        currentSpeedEl.textContent = payload.velocidad.toFixed(1);
        currentPlateEl.textContent = "BUSCANDO...";
        currentClassificationEl.textContent = "ANALIZANDO";
        plateCard.className = 'metric-card status-normal';
    } 
    else if (type === 'placa') {
        currentPlateEl.textContent = payload.placa;
    }
    else if (type === 'registro_guardado') {
        currentPlateEl.textContent = payload.placa;
        currentClassificationEl.textContent = payload.clasificacion.toUpperCase();
        
        if (payload.clasificacion === 'multa') {
            plateCard.className = 'metric-card status-multa';
            showToast(`ALERTA: Infracción detectada - ${payload.placa} a ${payload.velocidad} km/h`);
        } else {
            plateCard.className = 'metric-card status-normal';
        }
        
        addEventToList(payload);
    }
    else if (type === 'status') {
        if (payload.state === 'ready') {
            currentSpeedEl.textContent = "0.0";
            currentPlateEl.textContent = "---";
            currentClassificationEl.textContent = "ESPERANDO";
            plateCard.className = 'metric-card status-normal';
        } else if (payload.state === 'ended') {
            btnStop.click();
            showToast("Video finalizado.");
        }
    }
}

function addEventToList(payload) {
    const el = document.createElement('div');
    el.className = `event-item ${payload.clasificacion === 'multa' ? 'multa' : ''}`;
    
    el.innerHTML = `
        <div class="top">
            <span class="placa mono">${payload.placa}</span>
            <span class="badge">${payload.clasificacion.toUpperCase()}</span>
        </div>
        <span class="vel">${payload.velocidad} km/h</span>
    `;
    
    eventsListEl.prepend(el);
    if (eventsListEl.children.length > 10) {
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
