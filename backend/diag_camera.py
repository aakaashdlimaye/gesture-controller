"""
diag_camera.py - Diagnoses exactly what frames the camera delivers
in a background thread (identical conditions to gesture.py).
"""
import threading, time, os, sys

out_dir = os.path.dirname(os.path.abspath(__file__))

def camera_thread():
    import ctypes
    ret_com = ctypes.windll.ole32.CoInitializeEx(None, 0)
    print(f"CoInitializeEx returned: {ret_com:#010x}")  # 0 = S_OK, 1 = S_FALSE (already init'd)

    import cv2

    def probe(label, backend, extra_props=None):
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            print(f"[{label}] could not open")
            return
        if extra_props:
            for k, v in extra_props.items():
                cap.set(k, v)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fi  = int(cap.get(cv2.CAP_PROP_FOURCC))
        fcc = "".join(chr((fi >> 8*i) & 0xFF) for i in range(4))
        print(f"[{label}] w={w} h={h} fps={fps} fourcc={repr(fcc)}")

        # discard 20 warmup frames
        for _ in range(20):
            cap.read()

        frames = []
        t0 = time.monotonic()
        for _ in range(10):
            r, f = cap.read()
            if r:
                frames.append(f)
        elapsed = time.monotonic() - t0
        cap.release()

        if not frames:
            print(f"[{label}] NO frames after warmup")
            return

        f = frames[-1]
        mean_val = float(f.mean())
        std_val  = float(f.std())
        actual_fps = len(frames) / max(elapsed, 0.001)
        print(f"[{label}] shape={f.shape} dtype={f.dtype} "
              f"mean={mean_val:.1f} std={std_val:.1f} fps~={actual_fps:.1f}")

        path = os.path.join(out_dir, f"diag_{label}.jpg")
        saved = cv2.imwrite(path, f)
        print(f"[{label}] saved to {path}: {saved}")

    probe("DSHOW_default", cv2.CAP_DSHOW)
    probe("DSHOW_convert", cv2.CAP_DSHOW,
          {cv2.CAP_PROP_CONVERT_RGB: 1})
    probe("MSMF_default",  cv2.CAP_MSMF)

    ctypes.windll.ole32.CoUninitialize()
    print("done")

t = threading.Thread(target=camera_thread, daemon=False)
t.start()
t.join(timeout=90)
print("Diagnostic complete.")
