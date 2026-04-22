/**
 * app.js — Frontend controller for the Gesture Controller UI.
 *
 * Polls /status every 500ms and /logs every 1000ms.
 * No frameworks — plain ES2020.
 */

const API = "http://localhost:9000";

// DOM refs
const gestureName    = document.getElementById("gesture-name");
const confidenceFill = document.getElementById("confidence-fill");
const confidencePct  = document.getElementById("confidence-pct");
const activeApp      = document.getElementById("active-app");
const lastAction     = document.getElementById("last-action");
const webcamStatus   = document.getElementById("webcam-status");
const fpsCounter     = document.getElementById("fps-counter");
const logFeed        = document.getElementById("log-feed");
const startBtn       = document.getElementById("start-btn");
const stopBtn        = document.getElementById("stop-btn");
const connStatus     = document.getElementById("connection-status");
const modeBtns       = document.querySelectorAll(".mode-btn");
const videoFeed      = document.getElementById("video-feed");
const videoOverlay   = document.getElementById("video-overlay");

let statusInterval = null;
let logsInterval   = null;
let running        = false;

// ---------------------------------------------------------------------------
// Status polling
// ---------------------------------------------------------------------------

async function pollStatus() {
  try {
    const res  = await fetch(`${API}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateStatusUI(data);
    setConnected(true);
  } catch {
    setConnected(false);
  }
}

function updateStatusUI(data) {
  // Gesture name
  gestureName.textContent = data.gesture || "NONE";
  gestureName.classList.toggle("active", data.gesture !== "NONE");

  // Confidence bar
  const pct = Math.round((data.confidence || 0) * 100);
  confidenceFill.style.width = `${pct}%`;
  confidencePct.textContent  = `${pct}%`;
  confidenceFill.classList.remove("high", "mid", "low");
  if (pct >= 85)      confidenceFill.classList.add("high");
  else if (pct >= 50) confidenceFill.classList.add("mid");
  else                confidenceFill.classList.add("low");

  // Status fields
  activeApp.textContent    = data.active_app  || "unknown";
  lastAction.textContent   = data.last_action || "none";
  fpsCounter.textContent   = `${data.fps ?? "--"} FPS`;
  webcamStatus.textContent = data.webcam_active ? "active" : "inactive";
  webcamStatus.style.color = data.webcam_active
    ? "var(--green)" : "var(--red)";

  // Mode buttons
  modeBtns.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === data.mode);
  });
}

// ---------------------------------------------------------------------------
// Log polling
// ---------------------------------------------------------------------------

async function pollLogs() {
  try {
    const res  = await fetch(`${API}/logs`);
    if (!res.ok) return;
    const { logs } = await res.json();
    renderLogs(logs);
  } catch { /* ignore */ }
}

function renderLogs(logs) {
  if (!logs || logs.length === 0) {
    logFeed.innerHTML = '<div class="log-empty">No actions yet...</div>';
    return;
  }
  logFeed.innerHTML = logs
    .slice(0, 10)
    .map(entry => `<div class="log-entry">${escapeHtml(entry)}</div>`)
    .join("");
  logFeed.scrollTop = 0; // newest at top
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Connection indicator
// ---------------------------------------------------------------------------

function setConnected(ok) {
  connStatus.textContent = ok ? "Connected" : "Disconnected";
  connStatus.className   = `connection-status ${ok ? "connected" : "disconnected"}`;
}

// ---------------------------------------------------------------------------
// Start / Stop
// ---------------------------------------------------------------------------

function startVideoFeed() {
  // Setting src starts the MJPEG stream; append timestamp to bust any cache
  videoFeed.src = `${API}/video_feed?t=${Date.now()}`;
  videoFeed.style.display = "block";
  videoOverlay.classList.add("hidden");
}

function stopVideoFeed() {
  videoFeed.src = "";
  videoFeed.style.display = "none";
  videoOverlay.textContent = "Camera inactive — click Start";
  videoOverlay.classList.remove("hidden");
}

startBtn.addEventListener("click", async () => {
  try {
    const res = await fetch(`${API}/start`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    statusInterval = setInterval(pollStatus, 500);
    logsInterval   = setInterval(pollLogs, 1000);
    setConnected(true);
    startVideoFeed();
  } catch {
    setConnected(false);
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API}/stop`, { method: "POST" });
  } catch { /* best-effort */ }
  running = false;
  clearInterval(statusInterval);
  clearInterval(logsInterval);
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  gestureName.textContent = "NONE";
  confidenceFill.style.width = "0%";
  confidencePct.textContent = "0%";
  fpsCounter.textContent = "-- FPS";
  webcamStatus.textContent = "inactive";
  webcamStatus.style.color = "";
  stopVideoFeed();
});

// ---------------------------------------------------------------------------
// Mode buttons
// ---------------------------------------------------------------------------

modeBtns.forEach(btn => {
  btn.addEventListener("click", async () => {
    try {
      const res = await fetch(`${API}/mode/${btn.dataset.mode}`, { method: "POST" });
      if (res.ok) {
        modeBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
      }
    } catch { /* ignore */ }
  });
});

// ---------------------------------------------------------------------------
// Init — try to reach the backend immediately to show connection state
// ---------------------------------------------------------------------------

(async () => {
  try {
    const res  = await fetch(`${API}/status`);
    const data = await res.json();
    updateStatusUI(data);
    setConnected(true);
    // If backend already running, begin polling and show feed
    if (data.webcam_active) {
      running = true;
      startBtn.disabled = true;
      stopBtn.disabled  = false;
      statusInterval = setInterval(pollStatus, 500);
      logsInterval   = setInterval(pollLogs,   1000);
      startVideoFeed();
    }
  } catch {
    setConnected(false);
  }
})();
