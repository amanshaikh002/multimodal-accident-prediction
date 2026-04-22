"""
pose_service.py
===============
Core Worker Pose Safety detection service.

Pipeline
--------
1. Load YOLOv8s-pose model once (module-level singleton).
2. Load the trained ML classifier (model.pkl) once.
3. Open the video with cv2.VideoCapture.
4. Iterate through frames, processing every Nth frame (FRAME_STRIDE).
5. Resize each frame to INFER_W × INFER_H for faster inference.
6. Run YOLO pose inference (GPU if available).
7. Select the primary person (largest bbox + closest to centre).
8. Validate pose (essential joints above confidence threshold).
9. Extract ergonomic features via the shared utils module:
     - joint angles (back, knee, neck, elbow) — vector-based
     - temporal features (velocity, acceleration)
     - normalised keypoint coordinates (hip-centred, torso-scaled)
10. Classify posture with the ML model.
11. Apply 5-frame majority-vote temporal smoothing.
12. Accumulate SAFE / UNSAFE counts and violation log.
13. Optionally write annotated output video with skeleton + HUD overlay.
14. Return structured JSON summary.
"""

import logging
import math
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Imports — all from backend-local modules (no root-level dependencies)
# ---------------------------------------------------------------------------

from utils.pose_utils import (          # backend/utils/pose_utils.py
    FEATURE_COLS,
    LABEL_COLORS_BGR,
    LABEL_NAMES,
    SKELETON,
    build_feature_vector,
    build_violation_reason as _build_violation_reason,
    draw_hud_overlay       as _draw_hud_overlay_util,
    draw_skeleton          as _draw_skeleton_util,
    extract_all_features,
    is_pose_valid,
    make_empty_buffers,
    select_primary_person,
)
from utils.video_utils import get_video_writer, open_video   # backend/utils/video_utils.py

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_YOLO_MODEL_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "yolov8s-pose.pt")
)
_MODEL_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
)

_CONF_THRESH:   float = 0.25   # YOLO keypoint detection confidence floor
_FRAME_STRIDE:  int   = 2      # process every Nth frame (skip the rest)
_INFER_W:       int   = 640    # resize width  before inference
_INFER_H:       int   = 480    # resize height before inference
_SMOOTH_WINDOW: int   = 5      # majority-vote window (frames)
_SAVE_VIDEO:    bool  = True   # write annotated output video?

# Ergonomic thresholds (used for violation reason strings)
_BACK_UNSAFE:  float = 40.0   # degrees — back angle above this → unsafe
_BACK_MOD:     float = 20.0   # degrees — back angle above this → moderate
_KNEE_UNSAFE:  float = 100.0  # degrees — knee angle below this → unsafe
_KNEE_MOD:     float = 140.0  # degrees — knee angle below this → moderate
_NECK_UNSAFE:  float = 130.0  # degrees — neck angle below this → unsafe
_NECK_MOD:     float = 155.0  # degrees — neck angle below this → moderate

# Use GPU 0 if available, else CPU
_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_yolo_model:       Optional[YOLO] = None
_pose_classifier:  Optional[Any]  = None


def _get_yolo() -> YOLO:
    """Return the cached YOLO pose model, loading it on first call."""
    global _yolo_model
    if _yolo_model is None:
        # Use local model file if present, otherwise fall back to auto-download
        model_src = _YOLO_MODEL_PATH if os.path.isfile(_YOLO_MODEL_PATH) else "yolov8s-pose.pt"
        logger.info("Loading YOLO pose model: %s  (device=%s)", model_src, _DEVICE)
        _yolo_model = YOLO(model_src)
        logger.info("YOLO pose model loaded.")
    return _yolo_model


def _get_classifier() -> Any:
    """Return the cached ML classifier, loading it on first call."""
    global _pose_classifier
    if _pose_classifier is None:
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"Pose classifier not found at '{_MODEL_PATH}'. "
                "Run dataset_generator_yolo.py → auto_label.py → train_model.py first."
            )
        logger.info("Loading pose classifier from: %s", _MODEL_PATH)
        _pose_classifier = joblib.load(_MODEL_PATH)
        expected = getattr(_pose_classifier, "n_features_in_", len(FEATURE_COLS))
        if expected != len(FEATURE_COLS):
            logger.warning(
                "Classifier expects %d features but pipeline provides %d. "
                "Consider retraining the model.",
                expected, len(FEATURE_COLS),
            )
        logger.info("Pose classifier loaded — feature count: %d", len(FEATURE_COLS))
    return _pose_classifier


# Pre-load both models at import time (surface missing-file errors early)
try:
    _get_yolo()
except Exception as _err:
    logger.warning("YOLO pose model pre-load skipped: %s", _err)

try:
    _get_classifier()
except FileNotFoundError as _err:
    logger.warning("Pose classifier pre-load skipped: %s", _err)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_frame(
    classifier: Any,
    features: Dict[str, float],
) -> Tuple[int, float]:
    """
    Run the ML classifier on a single frame's feature vector.

    Returns
    -------
    (raw_label_int, confidence_float)
    """
    x_vec = build_feature_vector(features)

    if np.isnan(x_vec).any():
        return -1, 0.0

    raw_pred = int(classifier.predict(x_vec)[0])
    confidence = 0.0

    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(x_vec)[0]
        if raw_pred < len(proba):
            confidence = float(proba[raw_pred])
        else:
            confidence = float(proba.max())

    return raw_pred, confidence


def _build_violation_reason(features: Dict[str, float]) -> str:
    """
    Produce a concise, human-readable description of the worst ergonomic issue.
    Used as the "issue" field in the violations list.
    """
    reasons: List[str] = []

    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    if not math.isnan(ba):
        if ba > _BACK_UNSAFE:
            reasons.append("Excessive back bending")
        elif ba > _BACK_MOD:
            reasons.append("Moderate back lean")

    if not math.isnan(ka):
        if ka < _KNEE_UNSAFE:
            reasons.append("Deep knee bend — use support")
        elif ka < _KNEE_MOD:
            reasons.append("Moderate knee flexion")

    if not math.isnan(na):
        if na < _NECK_UNSAFE:
            reasons.append("Significant neck tilt")
        elif na < _NECK_MOD:
            reasons.append("Slight neck forward")

    return " | ".join(reasons) if reasons else "Bad lifting posture"


def _draw_skeleton(
    frame: np.ndarray,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_thr: float = 0.30,
) -> None:
    """Delegate to pose_utils.draw_skeleton."""
    _draw_skeleton_util(frame, kps_xy, kps_conf, conf_thr)


def _draw_hud_overlay(
    frame: np.ndarray,
    features: Dict[str, float],
    label: str,
    confidence: float,
    box_xyxy: np.ndarray,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Delegate to pose_utils.draw_hud_overlay."""
    _draw_hud_overlay_util(frame, features, label, confidence, box_xyxy, color_bgr, _INFER_H)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pose_video(
    video_path: str,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse a video for ergonomic / pose safety.

    Parameters
    ----------
    video_path        : absolute or relative path to the input video file
    output_video_path : if given (and _SAVE_VIDEO is True), an annotated
                        video is written to this path

    Returns
    -------
    dict – structured JSON-serialisable result:
    {
        "total_frames":  int,
        "safe_frames":   int,
        "unsafe_frames": int,
        "safety_score":  float,   # safe_frames / total_frames × 100
        "violations": [
            {"frame": int, "issue": str}
        ]
    }

    Raises
    ------
    FileNotFoundError  : video or model.pkl not found
    ValueError         : video is invalid / cannot be opened
    RuntimeError       : no person detected in ANY frame
    """
    # --- Load models (singleton — fast after first call) ---
    yolo   = _get_yolo()
    clf    = _get_classifier()

    # --- Open video ---
    cap = open_video(video_path)

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_raw  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # --- Optional annotated output ---
    writer: Optional[cv2.VideoWriter] = None
    if output_video_path and _SAVE_VIDEO:
        try:
            writer = get_video_writer(
                output_video_path,
                fps    = max(1.0, fps / _FRAME_STRIDE),
                width  = _INFER_W,
                height = _INFER_H,
            )
            logger.info("Annotated pose output → %s", output_video_path)
        except RuntimeError as exc:
            logger.warning("Cannot create VideoWriter: %s", exc)

    # --- Per-frame state ---
    buffers      = make_empty_buffers()                 # SmoothingBuffers for angles
    recent_preds: deque = deque(maxlen=_SMOOTH_WINDOW)  # majority-vote window

    safe_frames:   int = 0
    unsafe_frames: int = 0
    proc_frames:   int = 0   # successfully processed (pose found) frames
    raw_idx:       int = 0   # all frames read (before stride filtering)
    pose_detected: bool = False

    violations: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1

            # --- Frame stride: skip every other frame ---
            if raw_idx % _FRAME_STRIDE != 0:
                continue

            # --- Resize to inference resolution ---
            frame = cv2.resize(frame, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)

            # --- YOLO pose inference ---
            try:
                results = yolo.predict(
                    frame,
                    verbose=False,
                    conf=_CONF_THRESH,
                    imgsz=_INFER_W,
                    device=_DEVICE,
                )
            except Exception as exc:
                logger.warning("YOLO inference error on frame %d: %s", raw_idx, exc)
                _write_no_pose(frame, writer, "Inference error")
                continue

            result = results[0]

            no_pose = (
                result.keypoints is None
                or result.boxes is None
                or len(result.boxes) == 0
            )

            if no_pose:
                _write_no_pose(frame, writer, "No person detected")
                continue

            # --- Extract keypoints ---
            boxes        = result.boxes.xyxy.cpu().numpy()        # (N, 4)
            kps_xy_all   = result.keypoints.xy.cpu().numpy()      # (N, 17, 2)
            kps_conf_all = result.keypoints.conf.cpu().numpy()    # (N, 17)

            # --- Select primary person ---
            person_idx = select_primary_person(boxes, _INFER_W, _INFER_H)
            if person_idx is None:
                _write_no_pose(frame, writer, "No primary person")
                continue

            kps_xy    = kps_xy_all[person_idx]
            kps_conf  = kps_conf_all[person_idx]
            person_box = boxes[person_idx]

            # --- Validate pose keypoint quality ---
            if not is_pose_valid(kps_conf):
                _write_no_pose(frame, writer, "Low keypoint confidence")
                continue

            pose_detected = True  # At least one valid pose found

            # Draw skeleton before feature extraction (always visible if pose found)
            if writer is not None:
                _draw_skeleton(frame, kps_xy, kps_conf)

            # --- Feature extraction ---
            features = extract_all_features(kps_xy, kps_conf, buffers)

            if features is None:
                _write_no_pose(frame, writer, "Feature extraction failed")
                continue

            # --- Classify ---
            raw_pred, confidence = _classify_frame(clf, features)
            if raw_pred == -1:
                # NaN in feature vector — skip this frame
                continue

            # --- Temporal smoothing: 5-frame majority vote ---
            recent_preds.append(raw_pred)
            counter    = Counter(recent_preds)
            final_pred = counter.most_common(1)[0][0]

            label     = LABEL_NAMES.get(final_pred, "UNKNOWN")
            color_bgr = LABEL_COLORS_BGR.get(final_pred, (200, 200, 200))

            # --- Count safe / unsafe ---
            # Treat SAFE (0) and MODERATE (1) as safe frames; UNSAFE (2) as unsafe
            if final_pred == 2:       # UNSAFE
                unsafe_frames += 1
                reason = _build_violation_reason(features)
                violations.append({"frame": proc_frames, "issue": reason})
            else:                     # SAFE or MODERATE
                safe_frames += 1

            proc_frames += 1

            # --- Annotate frame (if writing video) ---
            if writer is not None:
                _draw_hud_overlay(frame, features, label, confidence, person_box, color_bgr)
                writer.write(frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # --- Guard: no person detected at all ---
    if not pose_detected:
        raise RuntimeError(
            "No person was detected in the video. "
            "Ensure the video contains a visible person in frame."
        )

    total_frames = safe_frames + unsafe_frames
    safety_score = round((safe_frames / total_frames) * 100, 1) if total_frames > 0 else 0.0

    summary = {
        "total_frames":  total_frames,
        "safe_frames":   safe_frames,
        "unsafe_frames": unsafe_frames,
        "safety_score":  safety_score,
        "violations":    violations,
    }

    logger.info(
        "Pose done — %d frames | safety %.1f%% | violations: %d",
        total_frames, safety_score, len(violations),
    )

    return summary


# ---------------------------------------------------------------------------
# Internal: write a 'no-pose' frame to the video writer
# ---------------------------------------------------------------------------

def _write_no_pose(
    frame: np.ndarray,
    writer: Optional[cv2.VideoWriter],
    msg: str,
) -> None:
    """Stamp a status message on the frame and write it to the output video."""
    cv2.putText(
        frame, msg, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA,
    )
    if writer is not None:
        writer.write(frame)
