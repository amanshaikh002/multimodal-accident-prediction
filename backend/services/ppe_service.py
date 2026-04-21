"""
ppe_service.py
==============
Core PPE detection service.

Flow
----
1. Load video with cv2.VideoCapture.
2. Iterate frames, skipping every odd frame (process every 2nd frame).
3. Resize each processed frame to (640, 480) before inference.
4. Run YOLOv8 inference.
5. Extract, filter (conf ≥ 0.5), and normalise detections.
6. Evaluate per-frame helmet / vest presence.
7. Optionally write an annotated output video.
8. Compute video-level compliance summary.
9. Return structured result dict ready to be serialised as JSON.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from utils.ppe_utils import (
    bbox_to_list,
    compute_compliance_metrics,
    compute_frame_ppe_status,
    draw_detections,
    filter_detections,
    normalize_label,
)
from utils.video_utils import cleanup_file, get_video_writer, open_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton — loaded once at module import time
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ppe_model.pt")
_MODEL_PATH = os.path.normpath(_MODEL_PATH)

_ppe_model: Optional[YOLO] = None


def _load_model() -> YOLO:
    """Load (or return cached) the YOLOv8 PPE model."""
    global _ppe_model
    if _ppe_model is None:
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"PPE model not found at '{_MODEL_PATH}'. "
                "Place your ppe_model.pt file inside backend/models/."
            )
        logger.info("Loading PPE model from: %s", _MODEL_PATH)
        _ppe_model = YOLO(_MODEL_PATH)
        logger.info("PPE model loaded successfully.")
    return _ppe_model


# Call at import time so the first request does not pay the load cost.
try:
    _load_model()
except FileNotFoundError as _e:
    logger.warning("PPE model not pre-loaded: %s", _e)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROCESS_WIDTH  = 640
_PROCESS_HEIGHT = 480
_CONF_THRESHOLD = 0.5
_FRAME_STRIDE   = 2          # process every 2nd frame
_SAVE_ANNOTATED = True       # set False to skip output video generation


def _run_inference_on_frame(
    model: YOLO,
    frame,
) -> List[Dict[str, Any]]:
    """
    Run YOLO inference on one frame and return raw detection dicts.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model.
    frame : np.ndarray
        BGR image of shape (H, W, 3).

    Returns
    -------
    list of dict
        Each dict: {label, confidence, bbox}
    """
    try:
        results = model(frame, verbose=False)
    except Exception as exc:
        logger.error("YOLO inference failed: %s", exc)
        return []

    raw_detections: List[Dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            try:
                label = result.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                bbox  = bbox_to_list(box.xyxy[0])
                raw_detections.append(
                    {"label": label, "confidence": round(conf, 4), "bbox": bbox}
                )
            except Exception as exc:
                logger.warning("Could not parse detection box: %s", exc)
                continue

    return raw_detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ppe_detection(
    video_path: str,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a video file for PPE compliance.

    Parameters
    ----------
    video_path : str
        Path to the input video (e.g. ``temp/temp.mp4``).
    output_video_path : str | None
        If provided, an annotated video is written here.
        Pass ``None`` to skip output video generation.

    Returns
    -------
    dict
        Structured result matching the API response schema::

            {
                "mode": "ppe",
                "summary": { ... },
                "frames": [ ... ]
            }

    Raises
    ------
    FileNotFoundError
        If the PPE model file cannot be found.
    ValueError
        If the video cannot be opened.
    """
    # --- Load model (uses cached singleton) ---
    model = _load_model()

    # --- Open video ---
    cap = open_video(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Optional: video writer for annotated output ---
    writer: Optional[cv2.VideoWriter] = None
    if output_video_path and _SAVE_ANNOTATED:
        try:
            writer = get_video_writer(
                output_video_path,
                fps=fps / _FRAME_STRIDE,   # corrected FPS after frame skipping
                width=_PROCESS_WIDTH,
                height=_PROCESS_HEIGHT,
            )
            logger.info("Annotated output will be saved to: %s", output_video_path)
        except RuntimeError as exc:
            logger.warning("Could not create VideoWriter: %s — skipping output video.", exc)

    # --- Frame processing loop ---
    frame_results: List[Dict[str, Any]] = []
    frame_index   = 0   # raw frame counter (all frames read)
    processed_id  = 0   # logical index for processed frames only

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            frame_index += 1

            # Skip odd frames (process every 2nd frame)
            if frame_index % _FRAME_STRIDE != 0:
                continue

            # Resize for inference
            resized = cv2.resize(frame, (_PROCESS_WIDTH, _PROCESS_HEIGHT))

            # Run YOLO inference
            raw_dets = _run_inference_on_frame(model, resized)

            # Filter by confidence + normalise labels
            filtered_dets = filter_detections(raw_dets, conf_threshold=_CONF_THRESHOLD)

            # Determine PPE status for this frame
            helmet_det, vest_det = compute_frame_ppe_status(filtered_dets)

            # Build per-frame result
            frame_record: Dict[str, Any] = {
                "frame_id":   processed_id,
                "helmet":     helmet_det,
                "vest":       vest_det,
                "detections": filtered_dets,
            }
            frame_results.append(frame_record)
            processed_id += 1

            # Optional: write annotated frame to output video
            if writer is not None:
                annotated = draw_detections(resized, filtered_dets, helmet_det, vest_det)
                writer.write(annotated)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # --- Compute video-level summary ---
    summary = compute_compliance_metrics(frame_results)

    logger.info(
        "PPE detection complete — %d frames processed | status: %s",
        summary["total_frames"],
        summary["status"],
    )

    return {
        "mode":    "ppe",
        "summary": summary,
        "frames":  frame_results,
    }
