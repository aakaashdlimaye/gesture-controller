"""Minimal timing test — no gesture import."""
import sys, time
import multiprocessing.connection as mpconn
t0 = time.monotonic()
def ts(m): print(f"{time.monotonic()-t0:.2f}s {m}", flush=True)
ts("start")
conn = mpconn.Client(("127.0.0.1", int(sys.argv[1])), authkey=b"test")
ts("connected")
import cv2
ts("cv2 imported")
cap = cv2.VideoCapture(0)
ts(f"VideoCapture isOpened={cap.isOpened()}")
ret, f = cap.read()
ts(f"read ret={ret}")
cap.release()
conn.close()
ts("done")
