"""
ppe_service.py
==============
Core PPE detection service.

Pipeline
--------
1. Load YOLOv8 model once (module-level singleton).
2. Open the video with cv2.VideoCapture.
3. Iterate through frames, processing every Nth frame (FRAME_STRIDE).
4. Resize each frame for faster inference.
5. Run inference with GPU if available.
6. Extract, normalise, and filter detections (confidence ≥ threshold).
7. Evaluate per-frame SAFE / UNSAFE status.
8. Optionally write an annotated output video.
9. Aggregate results into the required JSON summary.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO

from utils.ppe_utils import (
    bbox_to_list,
    compute_summary,
    draw_detections,
    evaluate_frame_safety,
    normalize_label,
)
from utils.video_utils import cleanup_file, get_video_writer, open_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_PATH    = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "ppe_model.pt")
)
_CONF_THRESH   = 0.45       # minimum confidence to keep a detection
_FRAME_STRIDE  = 2          # process every Nth frame (skip the rest)
_INFER_W       = 640        # resize width  before inference
_INFER_H       = 480        # resize height before inference
_SAVE_VIDEO    = True       # write annotated output video?

# Use GPU 0 if available, else CPU
_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    """Return the cached YOLO model, loading it on first call."""
    global _model
    if _model is None:
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"PPE model not found at '{_MODEL_PATH}'. "
                "Copy your ppe_model.pt into backend/models/."
            )
        logger.info("Loading PPE model from: %s  (device=%s)", _MODEL_PATH, _DEVICE)
        _model = YOLO(_MODEL_PATH)
        logger.info("PPE model loaded — classes: %s", list(_model.names.values()))
    return _model


# Pre-load at import time (catches missing model early)
try:
    _get_model()
except FileNotFoundError as _err:
    logger.warning("PPE model pre-load skipped: %s", _err)


# ---------------------------------------------------------------------------
# Internal: single-frame inference
# ---------------------------------------------------------------------------

def _infer_frame(model: YOLO, frame) -> List[Dict[str, Any]]:
    """
    Run YOLO on one frame and return a list of raw detection dicts.

    Each dict: {label, confidence, bbox}
    """
    try:
        results = model(frame, device=_DEVICE, verbose=False)
    except Exception as exc:
        logger.error("YOLO inference error: %s", exc)
        return []

    raw: List[Dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            try:
                label = result.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                bbox  = bbox_to_list(box.xyxy[0])
                raw.append({"label": label, "confidence": round(conf, 4), "bbox": bbox})
            except Exception as exc:
                logger.warning("Skipping malformed box: %s", exc)
    return raw


def _filter(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep detections above confidence threshold and normalise labels."""
    out = []
    for det in raw:
        if det["confidence"] < _CONF_THRESH:
            continue
        det["label"] = normalize_label(det["label"])
        out.append(det)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ppe_detection(
    video_path: str,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse a video for PPE compliance.

    Parameters
    ----------
    video_path        : path to input MP4
    output_video_path : if given, annotated video is written here

    Returns
    -------
    dict  – matches the required output JSON schema:
        {
            "total_frames":     int,
            "safe_frames":      int,
            "unsafe_frames":    int,
            "compliance_score": float,
            "violations": [
                {"frame": int, "missing": [str, ...]}
            ]
        }
    """
    model = _get_model()
    cap   = open_video(video_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer: Optional[cv2.VideoWriter] = None

    if output_video_path and _SAVE_VIDEO:
        try:
            writer = get_video_writer(
                output_video_path,
                fps    = max(1.0, fps / _FRAME_STRIDE),
                width  = _INFER_W,
                height = _INFER_H,
            )
            logger.info("Annotated output → %s", output_video_path)
        except RuntimeError as exc:
            logger.warning("Cannot create VideoWriter: %s", exc)

    frame_results: List[Dict[str, Any]] = []
    raw_idx  = 0    # all frames read counter
    proc_idx = 0    # processed-frame counter

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if raw_idx % _FRAME_STRIDE != 0:
                continue

            # ---- resize ----
            small = cv2.resize(frame, (_INFER_W, _INFER_H))

            # ---- infer + filter ----
            raw_dets  = _infer_frame(model, small)
            dets      = _filter(raw_dets)

            # ---- safety evaluation ----
            is_safe, missing = evaluate_frame_safety(dets)

            frame_results.append({
                "frame_id":   proc_idx,
                "safe":       is_safe,
                "missing":    missing,
                "detections": dets,
            })

            # ---- optional annotation ----
            if writer is not None:
                annotated = draw_detections(small, dets, is_safe, missing)
                writer.write(annotated)

            proc_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = compute_summary(frame_results)
    logger.info(
        "PPE done — %d frames | compliance %.1f%% | violations: %d",
        summary["total_frames"],
        summary["compliance_score"],
        len(summary["violations"]),
    )
    return summary
