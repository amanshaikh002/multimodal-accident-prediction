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

# Sticky-PPE smoothing — the trained PPE YOLO has poor recall on bent /
# crouched / non-frontal poses; it routinely loses the helmet or vest box
# for a handful of frames mid-motion. If a required item was detected on a
# visible worker within the last _PPE_STICKY_FRAMES processed frames, we
# treat it as still worn and don't flip to UNSAFE on a momentary miss.
# At stride 2 / 25 fps source ≈ 18 frames ≈ 1.5 s of grace per item.
# Set to 0 to disable smoothing.
_PPE_STICKY_FRAMES = 18

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
# Per-class confidence floor. Open-vocab grounding is unreliable on the
# "fallen person" prompt — a bending or crouching worker often matches at
# moderate confidence — so we hold it to a much higher bar than the static
# debris classes. Other prompts can stay permissive.
_HAZARD_CONF: Dict[str, float] = {
    "falling rock":   0.30,
    "loose brick":    0.30,
    "debris":         0.30,
    "fallen person":  0.55,
    "falling object": 0.35,
}
_HAZARD_CONF_DEFAULT: float = 0.30

# A genuinely fallen person's bbox is wider than tall. Bending, crouching,
# and squatting workers are still taller than wide. Requiring w/h >= this
# ratio kills the "bending worker tagged as fallen" false positive without
# needing any retraining.
_FALLEN_PERSON_MIN_AR: float = 1.2

# Single-frame YOLO-World hits are noisy. A label must persist for at least
# this many *consecutive* processed frames before it's treated as a real
# hazard. At stride 1 / source 25 fps this is ~0.16 s of evidence — fast
# enough to catch a falling brick, slow enough to discard flicker.
_HAZARD_PERSISTENCE_FRAMES: int = 2

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


def _passes_class_filters(label: str, conf: float, bbox: List[float]) -> bool:
    """
    Apply per-class confidence floor and shape constraints to a YOLO-World
    detection. Returns True if the detection should be kept.

    The "fallen person" prompt is gated extra hard:
      - much higher conf threshold than other hazard classes
      - bbox aspect ratio (w/h) must indicate a horizontal pose
    """
    floor = _HAZARD_CONF.get(label, _HAZARD_CONF_DEFAULT)
    if conf < floor:
        return False

    if label == "fallen person":
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        if (w / h) < _FALLEN_PERSON_MIN_AR:
            # Tall bbox → standing / bending / crouching worker, not fallen.
            return False

    return True


def _infer_hazards(frame) -> List[Dict[str, Any]]:
    """
    Run YOLO-World on one frame, returning hazard detections that pass the
    per-class filters. Returns [] if the hazard model is disabled or
    unavailable.

    NOTE: detections returned here are *candidates*; temporal persistence
    (consecutive-frame streak) is enforced at the call site.
    """
    model = _get_hazard_model()
    if model is None:
        return []

    # Use the lowest per-class floor as the model-side gate so we don't
    # discard borderline detections before our class-specific filter sees them.
    inference_floor = min(_HAZARD_CONF.values()) if _HAZARD_CONF else _HAZARD_CONF_DEFAULT

    try:
        results = model(frame, device=_DEVICE, conf=inference_floor, verbose=False)
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
                bbox  = bbox_to_list(box.xyxy[0])
                if not _passes_class_filters(label, conf, bbox):
                    continue
                raw.append({
                    "label":      label,
                    "confidence": round(conf, 4),
                    "bbox":       bbox,
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


def _confirm_hazards(
    candidates: List[Dict[str, Any]],
    streak: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Apply consecutive-frame persistence to candidate hazards.

    A hazard label only counts as confirmed once it's been seen in
    ``_HAZARD_PERSISTENCE_FRAMES`` consecutive processed frames. Labels
    that disappear reset their streak immediately.

    The ``streak`` dict is mutated in place: the caller owns it so that
    persistence carries across the loop without globals.

    Parameters
    ----------
    candidates : raw hazard detections from this frame (already class-filtered)
    streak     : {label: consecutive_frames_seen}, mutated in place

    Returns
    -------
    List of hazard detections whose label has reached the persistence
    threshold. The detection picked per label is the highest-confidence one
    from this frame, so banner / annotation reflect the strongest evidence.
    """
    seen_this_frame: Dict[str, Dict[str, Any]] = {}
    for det in candidates:
        label = det["label"]
        prev  = seen_this_frame.get(label)
        if prev is None or det["confidence"] > prev["confidence"]:
            seen_this_frame[label] = det

    # Bump streaks for labels seen, drop labels not seen this frame.
    for label in list(streak.keys()):
        if label not in seen_this_frame:
            del streak[label]
    for label in seen_this_frame:
        streak[label] = streak.get(label, 0) + 1

    confirmed: List[Dict[str, Any]] = []
    for label, det in seen_this_frame.items():
        if streak[label] >= _HAZARD_PERSISTENCE_FRAMES:
            confirmed.append(det)
    return confirmed


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

    # Per-label consecutive-frame counter — a label must persist for
    # _HAZARD_PERSISTENCE_FRAMES frames in a row before being treated as
    # a real hazard. Suppresses single-frame false positives.
    hazard_streak: Dict[str, int] = {}

    # Sticky-PPE state: last processed-frame index at which each required
    # PPE item was seen on any visible worker. -1 means "never seen yet".
    ppe_last_seen: Dict[str, int] = {"helmet": -1, "vest": -1}

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
            # Inference yields *candidates*; persistence enforces that a
            # hazard be seen in _HAZARD_PERSISTENCE_FRAMES consecutive frames
            # before being acted on. Cached candidates are reused on
            # stride-skipped frames so the streak logic still advances.
            if proc_idx % _HAZARD_STRIDE == 0:
                last_hazards = _infer_hazards(small)
            hazards = _confirm_hazards(last_hazards, hazard_streak)

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

            # ---- sticky-PPE smoothing ----
            # Update last-seen frame for every required PPE item visible now.
            for label in ("helmet", "vest"):
                if any(d["label"] == label for d in dets):
                    ppe_last_seen[label] = proc_idx

            ppe_smoothed = False
            if (
                _PPE_STICKY_FRAMES > 0
                and status == STATUS_UNSAFE
                and missing
                and any(d["label"] == "human" for d in dets)
            ):
                # Drop "missing" items that were seen on a worker in the
                # very recent past — most likely the model lost the box for
                # a frame or two while the worker bent / turned away.
                kept_missing: List[str] = []
                for item in missing:
                    last = ppe_last_seen.get(item, -1)
                    if last >= 0 and (proc_idx - last) <= _PPE_STICKY_FRAMES:
                        ppe_smoothed = True
                        continue
                    kept_missing.append(item)
                missing = sorted(kept_missing)
                if not missing:
                    status = STATUS_SAFE
                    reason = None

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
                "ppe_smoothed":  ppe_smoothed,
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
