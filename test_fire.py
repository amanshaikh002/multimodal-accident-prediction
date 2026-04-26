"""
test_fire.py
============
Diagnostic: run the YOLO fire-detection engine on the
first 150 frames of a video and print what it sees.

Usage:
    python test_fire.py path/to/video.mp4
"""
import sys
import cv2
from ultralytics import YOLO

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None
MODEL = "fire_best.pt"

print(f"\nLoading model: {MODEL}")
model = YOLO(MODEL)
print(f"Classes: {model.names}\n")

if VIDEO is None:
    print("No video path given.  Usage: python test_fire.py path/to/video.mp4")
    sys.exit(0)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open: {VIDEO}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video : {VIDEO}")
print(f"Frames: {total}   FPS: {fps:.1f}\n")

yolo_hits = 0
MAX = 150

for i in range(MAX):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # ── YOLO ────────────────────────────────────────────────────────────────
    res = model(frame, conf=0.01, iou=0.45, verbose=False)
    for r in res:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            lbl  = model.names.get(cls, str(cls))
            print(f"[YOLO] frame={i+1:>4} | {lbl:>8} | conf={conf:.3f}  {'← HIT' if conf>=0.20 else ''}")
            if conf >= 0.20:
                yolo_hits += 1

cap.release()

print(f"\n{'='*50}")
print(f"Scanned {min(MAX, total)} frames")
print(f"  YOLO hits (conf≥0.20) : {yolo_hits}")
if yolo_hits == 0:
    print("\n⚠  YOLO engine detected nothing.")
    print("   → The video may not contain visible fire, or the model needs retraining.")
