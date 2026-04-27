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
    STATUS_SAFE,
    STATUS_UNKNOWN,
    STATUS_UNSAFE,
    apply_hazard_override,
    bbox_to_list,
    compute_motion_score,
    compute_summary,
    draw_detections,
    evaluate_frame_safety,
    normalize_label,
)
from utils.video_utils import cleanup_file, finalize_video, get_video_writer, open_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_PATH    = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "ppe_model.pt")
)
# Class-specific confidence thresholds.
# Harnesses/vests are often missed by this model, so their threshold is very low.
_CLASS_THRESHOLDS = {
    "human":  0.30,
    "helmet": 0.20,
    "vest":   0.05,
    "gloves": 0.20,
    "boots":  0.20,
}
_FRAME_STRIDE  = 2          # process every Nth frame (skip the rest)
_INFER_W       = 640        # resize width  before inference
_INFER_H       = 480        # resize height before inference
_SAVE_VIDEO    = True       # write annotated output video?

# Occlusion handling — once a person has been seen, treat the next
# _OCCLUSION_GRACE_FRAMES processed frames without a person as UNKNOWN
# (not SAFE). At stride 2 / 25 fps source ≈ 25 frames ≈ 2 s of grace.
_OCCLUSION_GRACE_FRAMES = 25

# ---------------------------------------------------------------------------
# Open-vocabulary hazard detection (YOLO-World)
# ---------------------------------------------------------------------------
# Detects falling objects, debris, fallen workers, etc. without retraining.
# Override the model file via env var if you've cached a local copy; otherwise
# Ultralytics downloads `yolov8s-worldv2.pt` (~25 MB) on first use.
_HAZARD_MODEL_PATH: str   = os.environ.get("PPE_HAZARD_MODEL", "yolov8s-worldv2.pt")
_HAZARD_PROMPTS:   list   = [
    "falling rock",
    "loose brick",
    "debris",
    "fallen person",
    "falling object",
]
_HAZARD_CONF:      float  = 0.25     # confidence floor for hazard detections
_HAZARD_STRIDE:    int    = 1        # run hazard inference every N processed frames
_ENABLE_HAZARD:    bool   = os.environ.get("PPE_DISABLE_HAZARD", "").lower() not in ("1", "true", "yes")

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


# Hazard model singleton — loaded lazily, may fail in offline environments.
_hazard_model:        Optional[YOLO] = None
_hazard_load_failed:  bool           = False


def _get_hazard_model() -> Optional[YOLO]:
    """
    Return the cached YOLO-World model, or None if loading failed.

    The model is loaded on first call. If loading raises (e.g. no internet
    to download weights and no local file, or corrupt weights), we log once
    and return None on every subsequent call so the PPE pipeline degrades
    gracefully to the no-hazard path.
    """
    global _hazard_model, _hazard_load_failed
    if not _ENABLE_HAZARD or _hazard_load_failed:
        return None
    if _hazard_model is not None:
        return _hazard_model
    try:
        logger.info("Loading hazard model: %s  (device=%s)", _HAZARD_MODEL_PATH, _DEVICE)
        m = YOLO(_HAZARD_MODEL_PATH)
        m.set_classes(_HAZARD_PROMPTS)
        logger.info("Hazard model loaded — prompts: %s", _HAZARD_PROMPTS)
        _hazard_model = m
        return _hazard_model
    except Exception as exc:
        _hazard_load_failed = True
        logger.warning(
            "Hazard model unavailable (%s) — continuing without hazard detection. "
            "Set PPE_HAZARD_MODEL to a local path or PPE_DISABLE_HAZARD=1 to silence.",
            exc,
        )
        return None


def _infer_hazards(frame) -> List[Dict[str, Any]]:
    """
    Run YOLO-World on one frame, returning hazard detections above
    _HAZARD_CONF. Returns [] if the hazard model is disabled or unavailable.
    """
    model = _get_hazard_model()
    if model is None:
        return []
    try:
        results = model(frame, device=_DEVICE, conf=_HAZARD_CONF, verbose=False)
    except Exception as exc:
        logger.warning("Hazard inference error: %s", exc)
        return []

    raw: List[Dict[str, Any]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            try:
                label = result.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                if conf < _HAZARD_CONF:
                    continue
                raw.append({
                    "label":      label,
                    "confidence": round(conf, 4),
                    "bbox":       bbox_to_list(box.xyxy[0]),
                })
            except Exception as exc:
                logger.warning("Skipping malformed hazard box: %s", exc)
    return raw


# NOTE: Models are NOT pre-loaded at import time — loaded lazily on first request.


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
    """Keep detections above class-specific confidence threshold and normalise labels."""
    out = []
    for det in raw:
        norm_label = normalize_label(det["label"])
        thresh = _CLASS_THRESHOLDS.get(norm_label, 0.45)
        if det["confidence"] < thresh:
            continue
        det["label"] = norm_label
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

    # Occlusion-tracking state
    prev_gray: Optional[Any] = None
    ever_seen_person          = False
    frames_since_person_seen  = 10**9   # "never seen yet"

    # Hazard-state cache: when _HAZARD_STRIDE > 1 we reuse the most recent
    # hazard result on skipped frames so banners don't flicker.
    last_hazards: List[Dict[str, Any]] = []

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
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # ---- infer + filter ----
            raw_dets  = _infer_frame(model, small)
            dets      = _filter(raw_dets)

            # ---- open-vocab hazard inference (debris, falling rock, …) ----
            if proc_idx % _HAZARD_STRIDE == 0:
                last_hazards = _infer_hazards(small)
            hazards = last_hazards

            # ---- motion + recency context ----
            motion = compute_motion_score(prev_gray, gray)
            person_recently_seen = (
                ever_seen_person
                and frames_since_person_seen <= _OCCLUSION_GRACE_FRAMES
            )

            # ---- tri-state safety evaluation ----
            status, missing, reason = evaluate_frame_safety(
                dets,
                motion_score         = motion,
                person_recently_seen = person_recently_seen,
            )

            # ---- hazard override: forces UNSAFE if any hazard present ----
            status, missing, reason, hazard_labels = apply_hazard_override(
                status, missing, reason, hazards,
            )

            # ---- update rolling person-seen state ----
            if any(d["label"] == "human" for d in dets):
                ever_seen_person         = True
                frames_since_person_seen = 0
            else:
                frames_since_person_seen += 1

            frame_results.append({
                "frame_id":      proc_idx,
                "status":        status,
                "safe":          status == STATUS_SAFE,   # legacy field
                "missing":       missing,
                "reason":        reason,
                "motion":        round(motion, 4),
                "detections":    dets,
                "hazards":       hazard_labels,
                "hazard_boxes":  hazards,
            })

            # ---- optional annotation ----
            if writer is not None:
                annotated = draw_detections(
                    small, dets, status, missing, reason,
                    hazard_detections = hazards,
                )
                writer.write(annotated)

            prev_gray  = gray
            proc_idx  += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            if output_video_path:
                finalize_video(output_video_path)

    summary = compute_summary(frame_results)
    logger.info(
        "PPE done — %d frames | safe %d | unsafe %d | unknown %d | hazard %d | compliance %.1f%% | violations: %d",
        summary["total_frames"],
        summary["safe_frames"],
        summary["unsafe_frames"],
        summary["unknown_frames"],
        summary["hazard_frames"],
        summary["compliance_score"],
        len(summary["violations"]),
    )
    return summary
