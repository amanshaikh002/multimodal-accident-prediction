"""
fire_service.py
===============
Fire detection for the Industrial Safety System using YOLOv8.

Detection pipeline (per frame)
-------------------------------
1. YOLO engine  : run fire_best.pt at conf=0.05; accept boxes ≥ 0.20.
2. The frame is flagged as fire-positive if the YOLO engine triggers.

Returns
-------
{
    "module":        "fire",
    "status":        "UNSAFE" | "SAFE",
    "fire_detected": bool,
    "fire_ratio":    float,
    "total_frames":  int,
    "fire_frames":   int,
    "message":       str,
    "output_video":  str
}
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

from utils.video_utils import open_video, get_video_writer, finalize_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent.parent          # …/backend/
_MODEL_PATH  = str(_HERE.parent / "fire_best.pt")             # project root
_OUTPUT_DIR  = _HERE / "temp" / "output"
_OUTPUT_FILE = str(_OUTPUT_DIR / "fire_annotated.mp4")

# ── YOLO engine settings ────────────────────────────────────────────────────
_YOLO_CONF    = 0.05   # NMS conf passed to YOLO (very low – log everything)
_ACCEPT_CONF  = 0.20   # our own gate: only mark fire if conf ≥ this
_IOU_THRESH   = 0.45

# ── General ──────────────────────────────────────────────────────────────────
_FRAME_STRIDE = 1       # process EVERY frame
_INFER_W      = 640
_INFER_H      = 480

_FIRE_RATIO_THRESHOLD = 0.01   # any fire in ≥ 1% of frames → UNSAFE

_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    """Return the cached YOLO fire model, loading it on first call."""
    global _model
    if _model is None:
        if not os.path.isfile(_MODEL_PATH):
            _fallback = os.path.join(os.getcwd(), "fire_best.pt")
            src = _fallback if os.path.isfile(_fallback) else None
            if src is None:
                raise FileNotFoundError(
                    f"Fire model not found at '{_MODEL_PATH}'. "
                    "Place fire_best.pt in the project root."
                )
        else:
            src = _MODEL_PATH
        logger.info("[FIRE] Loading model from %s  (device=%s)", src, _DEVICE)
        _model = YOLO(src)
        logger.info("[FIRE] Model loaded — classes: %s", _model.names)
        print(f"[FIRE DEBUG] Model classes: {_model.names}")
    return _model


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_C_RED   = (0,   0, 255)
_C_WHITE = (255, 255, 255)
_C_GREEN = (0, 210,   0)


def _draw_box(
    frame: "cv2.Mat",
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw a filled-label bounding box."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    label_y = max(y1 - 6, lh + 6)
    cv2.rectangle(frame, (x1, label_y - lh - 6), (x1 + lw + 10, label_y + 2), color, -1)
    cv2.putText(frame, label, (x1 + 5, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, _C_WHITE, 2, cv2.LINE_AA)


def _draw_fire_banner(frame: "cv2.Mat") -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), _C_RED, -1)
    cv2.line(frame, (0, 50), (w, 50), _C_WHITE, 1)
    cv2.putText(frame, "  FIRE DETECTED — EVACUATE IMMEDIATELY",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, _C_WHITE, 2, cv2.LINE_AA)


def _draw_safe_banner(frame: "cv2.Mat") -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), _C_GREEN, -1)
    cv2.line(frame, (0, 32), (w, 32), _C_WHITE, 1)
    cv2.putText(frame, "  NO FIRE DETECTED",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, _C_WHITE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detection engines
# ---------------------------------------------------------------------------

def _yolo_detect(model: YOLO, frame: "cv2.Mat", frame_idx: int) -> bool:
    """
    Run YOLO inference and draw boxes.
    Returns True if fire was detected above _ACCEPT_CONF.
    """
    try:
        results = model(frame, conf=_YOLO_CONF, iou=_IOU_THRESH,
                        device=_DEVICE, verbose=False)
    except Exception as exc:
        logger.warning("[FIRE] YOLO error on frame %d: %s", frame_idx, exc)
        return False

    fire_found = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = model.names.get(cls, str(cls))
            print(f"[YOLO] frame={frame_idx} | label={label!r} | conf={conf:.3f}")

            if label.lower() != "fire":
                continue
            if conf < _ACCEPT_CONF:
                print(f"[YOLO SKIP] conf={conf:.3f} < {_ACCEPT_CONF}")
                continue

            fire_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"[YOLO HIT]  conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")
            _draw_box(frame, x1, y1, x2, y2, f"FIRE {conf:.2f}", _C_RED)

    return fire_found

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_fire_video(
    video_path: str,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyse a video for fire hazards using YOLO."""
    model    = _get_model()
    out_path = output_video_path or _OUTPUT_FILE

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer: Optional[cv2.VideoWriter] = None
    total_frames = 0
    fire_frames  = 0
    raw_idx      = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if raw_idx % _FRAME_STRIDE != 0:
                continue

            total_frames += 1
            frame = cv2.resize(frame, (_INFER_W, _INFER_H),
                               interpolation=cv2.INTER_LINEAR)

            # ── YOLO Detection ────────────────────────────────────────────
            fire_this_frame = _yolo_detect(model, frame, raw_idx)

            if fire_this_frame:
                fire_frames += 1
                _draw_fire_banner(frame)
                logger.info("[FIRE] Frame %d — fire detected by YOLO", raw_idx)
            else:
                _draw_safe_banner(frame)

            # ── Init writer ───────────────────────────────────────────────
            if writer is None:
                try:
                    effective_fps = max(1.0, fps / _FRAME_STRIDE)
                    writer = get_video_writer(
                        out_path,
                        fps    = effective_fps,
                        width  = _INFER_W,
                        height = _INFER_H,
                    )
                except RuntimeError as exc:
                    logger.error("[FIRE] Cannot create VideoWriter: %s", exc)
                    break

            writer.write(frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            finalize_video(out_path)
            logger.info("[FIRE] Output finalized → %s", out_path)

    fire_ratio    = fire_frames / total_frames if total_frames > 0 else 0.0
    fire_detected = fire_frames > 0
    status        = "UNSAFE" if fire_ratio > _FIRE_RATIO_THRESHOLD else "SAFE"

    logger.info(
        "[FIRE] Done — %d frames | fire_frames=%d | ratio=%.3f | status=%s",
        total_frames, fire_frames, fire_ratio, status,
    )
    print(f"[FIRE SUMMARY] total={total_frames} | fire_frames={fire_frames} "
          f"| ratio={fire_ratio:.4f} | status={status}")

    message = (
        "Fire hazard detected! Evacuate immediately and contact emergency services."
        if fire_detected else
        "No fire detected. Workplace appears safe."
    )

    return {
        "module":        "fire",
        "status":        status,
        "fire_detected": fire_detected,
        "fire_ratio":    round(fire_ratio, 4),
        "total_frames":  total_frames,
        "fire_frames":   fire_frames,
        "message":       message,
        "output_video":  "fire_annotated.mp4",
    }
