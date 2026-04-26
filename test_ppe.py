import sys
import cv2
from ultralytics import YOLO

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "backend/temp/upload_ppe.mp4"
MODEL = "backend/models/ppe_model.pt"

print(f"\nLoading model: {MODEL}")
model = YOLO(MODEL)
print(f"Classes: {model.names}\n")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open: {VIDEO}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames: {total}, FPS: {fps}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_idx += 1
    if frame_idx % 30 != 0: continue
    
    res = model(frame, conf=0.01, verbose=False)
    print(f"\n--- Frame {frame_idx} ---")
    for r in res:
        if r.boxes is None: continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            lbl = model.names[cls]
            if conf >= 0.1: # Only print above 10% to reduce noise
                print(f"Detected: {lbl:>12} | Conf: {conf:.3f}")

cap.release()
