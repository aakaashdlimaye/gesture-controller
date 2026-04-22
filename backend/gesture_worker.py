"""
gesture_worker.py — Camera + gesture detection subprocess.

Uses OpenCV VideoCapture for camera capture.
Communicates with the server via a multiprocessing.connection TCP socket.

Usage: python gesture_worker.py <port> <authkey_hex>
"""

import logging
import os
import sys
import time
from collections import deque

import numpy as np

# Make sure gesture.py (in the same directory) is importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from gesture import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    GESTURE_INDEX_UP,
    GESTURE_NONE,
    classify_gesture,
)


def main(port: int, authkey: bytes) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [INFO] gesture.worker: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    import multiprocessing.connection as mpconn

    log.info(f"Connecting to server on 127.0.0.1:{port} ...")
    try:
        conn = mpconn.Client(("127.0.0.1", port), authkey=authkey)
    except Exception as exc:
        log.error(f"Failed to connect to server: {exc}")
        return
    log.info("Connected.")

    # ── Import cv2 ────────────────────────────────────────────────────────
    try:
        import cv2
    except ImportError as exc:
        log.error(f"Missing cv2: {exc}")
        _send(conn, {"type": "webcam_active", "value": False})
        conn.close()
        return

    # ── Open camera via OpenCV ────────────────────────────────────────────
    log.info("Opening camera with OpenCV...")
    cap = None
    for index in range(4):
        log.info(f"Trying camera index {index}...")
        c = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if c.isOpened():
            # Verify we can actually read a frame
            ok, frame = c.read()
            if ok and frame is not None and float(np.mean(frame)) > 1.0:
                log.info(f"Camera opened at index {index}")
                cap = c
                break
            c.release()
        else:
            c.release()

    if cap is None:
        # Fallback: try default backend
        for index in range(4):
            log.info(f"Trying camera index {index} (default backend)...")
            c = cv2.VideoCapture(index)
            if c.isOpened():
                ok, frame = c.read()
                if ok and frame is not None and float(np.mean(frame)) > 1.0:
                    log.info(f"Camera opened at index {index} (default backend)")
                    cap = c
                    break
                c.release()
            else:
                c.release()

    if cap is None:
        log.error("Could not open any camera.")
        _send(conn, {"type": "webcam_active", "value": False})
        conn.close()
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Import MediaPipe ──────────────────────────────────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        log.error(f"Missing mediapipe: {exc}")
        cap.release()
        _send(conn, {"type": "webcam_active", "value": False})
        conn.close()
        return

    model_path = os.path.join(_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        log.error(f"hand_landmarker.task not found at {model_path}")
        cap.release()
        _send(conn, {"type": "webcam_active", "value": False})
        conn.close()
        return

    log.info("Loading MediaPipe model...")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    log.info("MediaPipe model loaded. Streaming...")

    _send(conn, {"type": "webcam_active", "value": True})

    wrist_history: deque = deque(maxlen=8)
    last_swipe_time: float = 0.0
    prev_time = time.monotonic()
    fps_ema: float = 0.0

    try:
        while True:
            # Check stop signal (non-blocking)
            try:
                if conn.poll(0) and conn.recv() == "stop":
                    break
            except Exception:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Frame read failed, retrying...")
                time.sleep(0.05)
                continue

            now = time.monotonic()
            delta = max(now - prev_time, 1e-9)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect_for_video(mp_image, int(now * 1000))

            gesture_name = GESTURE_NONE
            confidence = 0.0
            cursor_x = cursor_y = 0.0

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                gesture_name, confidence, last_swipe_time = classify_gesture(
                    lm, wrist_history, now, last_swipe_time
                )
                if gesture_name == GESTURE_INDEX_UP:
                    cursor_x, cursor_y = lm[8].x, lm[8].y

                h, w = frame.shape[:2]
                for pt in lm:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 0), -1)

            label = gesture_name if confidence >= CONFIDENCE_THRESHOLD else "NONE"
            color = (0, 255, 100) if confidence >= CONFIDENCE_THRESHOLD else (80, 80, 80)
            cv2.putText(frame, label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{int(confidence * 100)}%", (12, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            fps_ema = 0.9 * fps_ema + 0.1 / delta

            msg = {
                "type":         "state",
                "gesture_name": gesture_name if confidence >= CONFIDENCE_THRESHOLD else GESTURE_NONE,
                "confidence":   confidence,
                "timestamp":    now,
                "fps":          fps_ema,
                "cursor":       (cursor_x, cursor_y) if gesture_name == GESTURE_INDEX_UP else None,
                "frame":        buf.tobytes() if ok else None,
            }
            if not _send(conn, msg):
                break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        _send(conn, {"type": "webcam_active", "value": False})
        try:
            conn.close()
        except Exception:
            pass
        log.info("Worker exiting.")


def _send(conn, msg: dict) -> bool:
    try:
        conn.send(msg)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gesture_worker.py <port> <authkey_hex>")
        sys.exit(1)
    _port = int(sys.argv[1])
    _key = bytes.fromhex(sys.argv[2])
    main(_port, _key)
