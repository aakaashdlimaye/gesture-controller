"""
gesture.py — Gesture classification helpers and GestureDetector.

GestureDetector spawns gesture_worker.py as a subprocess (via subprocess.Popen)
so that cv2.VideoCapture runs in that process's main thread.
This is required on Windows because MSMF (the default OpenCV camera backend)
hangs when called from a background thread or a multiprocessing.Process child.
IPC is handled by multiprocessing.connection over a localhost TCP socket.
"""

import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from main import AppState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75

GESTURE_NONE        = "NONE"
GESTURE_PINCH       = "PINCH"
GESTURE_OPEN_PALM   = "OPEN_PALM"
GESTURE_FIST        = "FIST"
GESTURE_SWIPE_LEFT  = "SWIPE_LEFT"
GESTURE_SWIPE_RIGHT = "SWIPE_RIGHT"
GESTURE_INDEX_UP    = "INDEX_UP"


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

@dataclass
class _LM:
    """Minimal landmark stub used by unit tests (mirrors mediapipe NormalizedLandmark)."""
    x: float
    y: float
    z: float = 0.0


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _finger_extended(lm: list, tip: int, pip: int) -> bool:
    """True if fingertip is above PIP joint (smaller y = higher on screen)."""
    return lm[tip].y < lm[pip].y - 0.02


def _thumb_extended(lm: list) -> bool:
    """Thumb uses horizontal distance (it moves laterally, not vertically)."""
    return abs(lm[4].x - lm[2].x) > 0.08


def _open_palm_score(lm: list) -> float:
    """Fraction of fingers that are extended (0.0 – 1.0)."""
    extended = [
        _finger_extended(lm, 8,  6),
        _finger_extended(lm, 12, 10),
        _finger_extended(lm, 16, 14),
        _finger_extended(lm, 20, 18),
        _thumb_extended(lm),
    ]
    return sum(extended) / 5.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_gesture(
    lm: list,
    wrist_history: deque,
    now: float,
    last_swipe_time: float,
) -> Tuple[str, float, float]:
    """
    Classify hand gesture from 21 MediaPipe landmarks.
    Returns: (gesture_name, confidence, updated_last_swipe_time)
    """
    SWIPE_COOLDOWN        = 0.8
    SWIPE_VELOCITY_THRESH = 1.5
    PINCH_RATIO_THRESH    = 0.4

    wrist_history.append((now, lm[0].x))
    palm_score = _open_palm_score(lm)

    # 1. SWIPE (temporal, checked first)
    if now - last_swipe_time > SWIPE_COOLDOWN and len(wrist_history) >= 6:
        oldest_t, oldest_x = wrist_history[0]
        newest_t, newest_x = wrist_history[-1]
        time_delta = newest_t - oldest_t
        if time_delta > 0.05 and palm_score > 0.6:
            velocity = (newest_x - oldest_x) / time_delta
            if velocity > SWIPE_VELOCITY_THRESH:
                return GESTURE_SWIPE_RIGHT, min(abs(velocity) / 3.0, 1.0), now
            elif velocity < -SWIPE_VELOCITY_THRESH:
                return GESTURE_SWIPE_LEFT, min(abs(velocity) / 3.0, 1.0), now

    # 2. PINCH
    palm_size   = max(_dist(lm[0], lm[9]), 1e-6)
    pinch_ratio = _dist(lm[4], lm[8]) / palm_size
    if pinch_ratio < PINCH_RATIO_THRESH:
        return GESTURE_PINCH, max(0.0, 1.0 - (pinch_ratio / PINCH_RATIO_THRESH)), last_swipe_time

    # 3. INDEX_UP
    if (
        _finger_extended(lm, 8,  6)
        and not _finger_extended(lm, 12, 10)
        and not _finger_extended(lm, 16, 14)
        and not _finger_extended(lm, 20, 18)
    ):
        return GESTURE_INDEX_UP, 0.9, last_swipe_time

    # 4. OPEN_PALM
    if palm_score >= 0.8:
        return GESTURE_OPEN_PALM, palm_score, last_swipe_time

    # 5. FIST
    fist_parts = [
        not _finger_extended(lm, 8,  6),
        not _finger_extended(lm, 12, 10),
        not _finger_extended(lm, 16, 14),
        not _finger_extended(lm, 20, 18),
    ]
    fist_score = sum(fist_parts) / 4.0
    if fist_score >= 0.75:
        return GESTURE_FIST, fist_score, last_swipe_time

    return GESTURE_NONE, 0.0, last_swipe_time


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

_AUTHKEY = b"gesture-ipc-2024"
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_worker.py")


class GestureDetector:
    """
    Spawns gesture_worker.py as an independent subprocess (subprocess.Popen)
    so cv2.VideoCapture runs in the worker's own main thread.
    Results arrive via a multiprocessing.connection TCP socket.
    """

    def __init__(self, state: "AppState"):
        self.state = state
        self._proc:     Optional[subprocess.Popen] = None
        self._listener = None
        self._conn     = None
        self._reader:  Optional[threading.Thread]  = None

    def start(self):
        import logging
        import multiprocessing.connection as mpconn

        log = logging.getLogger(__name__)

        # Bind a listener on a random localhost port
        self._listener = mpconn.Listener(("127.0.0.1", 0), authkey=_AUTHKEY)
        port = self._listener.address[1]
        log.info(f"Gesture IPC listener on port {port}")

        # Start the worker as a fully independent subprocess.
        # stdin=DEVNULL prevents the worker (and its children) from
        # inheriting the server's stdin, which can cause ffmpeg to exit.
        self._proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT, str(port), _AUTHKEY.hex()],
            stdin=subprocess.DEVNULL,
        )

        # Accept the worker's connection (timeout 30 s)
        self._listener._listener._socket.settimeout(30.0)
        try:
            self._conn = self._listener.accept()
        except Exception as exc:
            log.error(f"Gesture worker did not connect within 30 s: {exc}")
            self._proc.terminate()
            return

        log.info("Gesture worker connected.")

        # Start the reader thread
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="GestureReader"
        )
        self._reader.start()

    def stop(self):
        # Signal worker to stop
        try:
            if self._conn:
                self._conn.send("stop")
        except Exception:
            pass
        # Wait then force-terminate
        if self._proc:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        # Close IPC
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        try:
            if self._listener:
                self._listener.close()
        except Exception:
            pass

    def _read_loop(self):
        """Blocks on conn.recv(); updates AppState for every message received."""
        import logging
        log = logging.getLogger(__name__)

        while not self.state.stop_event.is_set():
            try:
                msg = self._conn.recv()
            except Exception:
                break  # connection closed or worker crashed

            mtype = msg.get("type")

            if mtype == "webcam_active":
                with self.state.lock:
                    self.state.webcam_active = msg["value"]

            elif mtype == "state":
                with self.state.lock:
                    self.state.gesture.name       = msg["gesture_name"]
                    self.state.gesture.confidence = msg["confidence"]
                    self.state.gesture.timestamp  = msg["timestamp"]
                    self.state.fps                = msg["fps"]
                    if msg.get("cursor") is not None:
                        self.state.cursor_target  = msg["cursor"]
                if msg.get("frame"):
                    with self.state.frame_lock:
                        self.state.latest_frame = msg["frame"]

        with self.state.lock:
            self.state.webcam_active = False
        log.info("Gesture reader thread exiting.")
