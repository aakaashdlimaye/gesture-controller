"""Timing diagnostic for gesture worker camera init."""
import time
import sys
import os

t0 = time.monotonic()

def ts(msg):
    print(f"{time.monotonic()-t0:.2f}s {msg}", flush=True)

ts("start")
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ts("stdlib ready")
from gesture import CONFIDENCE_THRESHOLD, GESTURE_NONE, classify_gesture
ts("gesture imported")
import multiprocessing.connection as mpconn
ts("mpconn imported")

port = int(sys.argv[1])
conn = mpconn.Client(("127.0.0.1", port), authkey=b"test")
ts("connected to listener")

import cv2
ts("cv2 imported")
cap = cv2.VideoCapture(0)
ts(f"VideoCapture created, isOpened={cap.isOpened()}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
ts("props set")
ret, f = cap.read()
ts(f"read: ret={ret} shape={f.shape if ret else None}")
cap.release()
conn.close()
ts("done")
