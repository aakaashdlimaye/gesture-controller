"""
main.py — FastAPI server and shared application state.

Run with:  python main.py
Or:        uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Load .env before any module imports that might read env vars
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

@dataclass
class GestureState:
    name: str = "NONE"
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class AppState:
    # Gesture thread writes these
    gesture: GestureState = field(default_factory=GestureState)
    fps: float = 0.0
    webcam_active: bool = False
    cursor_target: tuple = field(default_factory=lambda: (0.0, 0.0))

    # Agent thread writes these
    active_app: str = "unknown"
    last_action: str = "none"
    current_mode: str = "auto"  # auto | volume | slides | cursor

    # Ring buffer — newest entries at index 0 (appendleft)
    action_log: deque = field(default_factory=lambda: deque(maxlen=50))

    # Latest JPEG frame bytes for /video_feed (written by gesture thread)
    latest_frame: bytes = field(default_factory=bytes)
    frame_lock: threading.Lock = field(default_factory=threading.Lock)

    # Lifecycle
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


# Singleton state shared by all modules
state = AppState()

# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

_gesture_detector = None
_agent_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Server is running
    # Graceful shutdown
    logger.info("Shutting down — stopping gesture and agent threads...")
    state.stop_event.set()
    if _gesture_detector:
        _gesture_detector.stop()
    if _agent_engine and _agent_engine._thread:
        _agent_engine._thread.join(timeout=3.0)
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(title="Gesture Controller", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

VALID_MODES = {"auto", "volume", "slides", "cursor"}



@app.get("/status")
def get_status() -> dict:
    """Return current system state for the frontend to poll."""
    with state.lock:
        return {
            "gesture":       state.gesture.name,
            "confidence":    round(state.gesture.confidence, 3),
            "active_app":    state.active_app,
            "last_action":   state.last_action,
            "fps":           round(state.fps, 1),
            "mode":          state.current_mode,
            "webcam_active": state.webcam_active,
        }


@app.post("/start")
def start() -> dict:
    """Start webcam capture and the gesture/agent threads."""
    global _gesture_detector, _agent_engine

    with state.lock:
        already = state.webcam_active

    if already:
        return {"status": "already_running"}

    # Stop any previous gesture detector (kills child process + reader thread)
    if _gesture_detector:
        logger.info("Stopping previous gesture detector...")
        _gesture_detector.stop()

    # Import here to allow the server to start even if mediapipe is missing
    from gesture import GestureDetector
    from agent import AgentEngine

    state.stop_event.clear()

    _gesture_detector = GestureDetector(state)
    _agent_engine = AgentEngine(state)

    _gesture_detector.start()
    _agent_engine.start()

    logger.info("Gesture detection and agent engine started.")
    return {"status": "started"}


@app.post("/stop")
def stop() -> dict:
    """Stop webcam capture and gesture/agent threads."""
    state.stop_event.set()
    logger.info("Stop requested.")
    return {"status": "stopped"}


@app.post("/mode/{mode}")
def set_mode(mode: str) -> dict:
    """Switch operating mode: auto | volume | slides | cursor."""
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Valid modes: {sorted(VALID_MODES)}",
        )
    with state.lock:
        state.current_mode = mode
    logger.info(f"Mode switched to '{mode}'.")
    return {"mode": mode}


@app.get("/logs")
def get_logs() -> dict:
    """Return last 50 action log entries (newest first)."""
    with state.lock:
        return {"logs": list(state.action_log)[:50]}


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the webcam with gesture overlay."""
    async def generate():
        while True:
            with state.frame_lock:
                frame = state.latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await __import__("asyncio").sleep(0.033)  # ~30 fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    # Use SelectorEventLoop on Windows — ProactorEventLoop's IOCP interferes
    # with DSHOW camera access in child subprocesses.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
        log_level="info",
        loop="asyncio",
    )
