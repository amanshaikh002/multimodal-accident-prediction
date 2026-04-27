"""
fire_service.py
===============
Fire detection for the Industrial Safety System.

Default detector
----------------
A dedicated YOLOv8 fire/smoke model trained on the D-Fire dataset, pulled
from HuggingFace on first use:
``arnabdhar/YOLOv8-Fire-and-Smoke-Detection`` (override via FIRE_HF_REPO).
A custom YOLO fire model can still be plugged in via the ``FIRE_MODEL`` env
var. YOLO-World remains as a last-resort fallback if the HuggingFace
download fails (e.g. offline environments).

Loader priority
---------------
1. ``FIRE_MODEL`` env var (path to a local YOLO weights file)
2. HuggingFace fire model (auto-downloaded once, cached)
3. YOLO-World prompted with fire vocabulary (fallback)

Per-frame pipeline
------------------
1. **Detect**   — YOLO inference, fire-related labels above _ACCEPT_CONF.
2. **Persist**  — A frame is fire-positive only after _FIRE_PERSISTENCE_FRAMES
                  consecutive raw detections (kills single-frame flicker).

HSV color verification is OFF by default. Set ``FIRE_HSV_VERIFY=1`` to
re-enable it as a secondary gate; it is no longer required.

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
import numpy as np
import torch
from ultralytics import YOLO

from utils.video_utils import open_video, get_video_writer, finalize_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent.parent          # …/backend/
_OUTPUT_DIR  = _HERE / "temp" / "output"
_OUTPUT_FILE = str(_OUTPUT_DIR / "fire_annotated.mp4")

# Custom fire model (optional). Priority order:
#   1. FIRE_MODEL env var, if it points at an existing file.
#      (The legacy fire_best.pt at the project root is NEVER auto-loaded —
#      set FIRE_MODEL to its full path if you ever want to revive it.)
#   2. HuggingFace fire model (auto-downloaded once, cached).
#   3. YOLO-World fallback prompted with fire vocabulary.
_CUSTOM_MODEL_ENV    = os.environ.get("FIRE_MODEL", "").strip()

# HuggingFace defaults — verified public general-purpose fire/smoke YOLO
# (SalahALHaismawi/yolov26-fire-detection, ~20 MB, classes: fire, other,
# smoke). Verified on the user's industrial fire video: detections land on
# the actual flames, not on floor reflections. Override via env vars if
# you want a different model.
_HF_REPO       = os.environ.get("FIRE_HF_REPO",     "SalahALHaismawi/yolov26-fire-detection")
_HF_FILENAME   = os.environ.get("FIRE_HF_FILENAME", "best.pt")

_WORLD_MODEL_PATH    = os.environ.get("FIRE_WORLD_MODEL", "yolov8s-worldv2.pt")
_WORLD_FIRE_PROMPTS  = ["fire", "flame", "flames", "burning"]

# Fire class labels we accept from any model (case-insensitive substring match).
_FIRE_LABEL_KEYWORDS = ("fire", "flame", "flames", "burning")

# ── YOLO engine settings ────────────────────────────────────────────────────
_YOLO_CONF    = 0.05   # NMS conf — log everything, gate at _ACCEPT_CONF
_ACCEPT_CONF  = 0.20   # gate on raw detection conf
_IOU_THRESH   = 0.45

# Per-detection logging. Set FIRE_VERBOSE=0 to silence the per-frame
# label/confidence dump. Default ON so it's easy to diagnose missed detections.
_VERBOSE_DETECTIONS = os.environ.get("FIRE_VERBOSE", "1").lower() not in ("0", "false", "no")

# ── HSV fire-color verification (OPT-IN) ────────────────────────────────────
# OFF by default. Enable by setting FIRE_HSV_VERIFY=1. Useful only when
# pairing with the YOLO-World fallback, which is more prone to false
# positives on bright lights / sunsets. The HF fire model is dedicated and
# does not need this gate.
_USE_HSV_VERIFICATION = os.environ.get("FIRE_HSV_VERIFY", "").lower() in ("1", "true", "yes")
_MIN_FIRE_PIXEL_RATIO = 0.05    # 5 % of box pixels must be fire-coloured
_HSV_LOW_1  = np.array([  0, 100, 150], dtype=np.uint8)   # red-orange
_HSV_HIGH_1 = np.array([ 35, 255, 255], dtype=np.uint8)
_HSV_LOW_2  = np.array([165, 100, 150], dtype=np.uint8)   # wrap-around red
_HSV_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)

# Single-frame flicker filter — fire must persist this many consecutive
# processed frames before we count the frame as fire-positive.
_FIRE_PERSISTENCE_FRAMES = 2

# ── General ──────────────────────────────────────────────────────────────────
_FRAME_STRIDE = 1       # process EVERY frame

# Inference image size (passed to YOLO as imgsz=...). YOLO letterboxes the
# original frame to this square, so portrait/landscape aspect is preserved.
# Larger imgsz → better detection of small/distant fires, slower inference.
# Default 1280 is a good speed/accuracy trade-off on modern GPUs; bump to
# 1536 if you have headroom, drop to 960 if you need faster CPU inference.
_INFER_IMGSZ  = int(os.environ.get("FIRE_INFER_IMGSZ", "1280"))

# Cap the output video's longest side so we don't write multi-GB files for
# 4K input. Output keeps the source aspect ratio. Set FIRE_OUTPUT_MAX_SIDE=0
# to keep the original size verbatim.
_OUTPUT_MAX_SIDE = int(os.environ.get("FIRE_OUTPUT_MAX_SIDE", "1280"))

_FIRE_RATIO_THRESHOLD = 0.01   # any fire in ≥ 1 % of confirmed frames → UNSAFE

_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model:        Optional[YOLO] = None
_model_kind:   str            = ""    # "custom" | "world"
_model_source: str            = ""    # path / weights name actually loaded


def _validate_fire_model(m: YOLO, source: str) -> bool:
    """Confirm a loaded YOLO model exposes at least one fire-ish class."""
    names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
    has_fire = any(any(kw in n.lower() for kw in _FIRE_LABEL_KEYWORDS) for n in names)
    if not has_fire:
        logger.warning("[FIRE] %s has no fire class (classes=%s)", source, names)
        return False
    logger.info("[FIRE] %s classes: %s", source, names)
    return True


def _try_load_custom() -> Optional[Tuple[YOLO, str]]:
    """
    Load a user-supplied custom YOLO fire model — only when FIRE_MODEL is
    explicitly set in the environment.

    The legacy ``fire_best.pt`` at the project root is intentionally NOT
    auto-loaded: it has been confirmed to miss fire on real footage. To
    revive it deliberately, set FIRE_MODEL=<full path to fire_best.pt>.
    """
    if not _CUSTOM_MODEL_ENV:
        return None

    path = _CUSTOM_MODEL_ENV
    if not os.path.isfile(path):
        logger.warning("[FIRE] FIRE_MODEL points at non-existent path: %s", path)
        return None

    try:
        logger.info("[FIRE] Loading custom fire model: %s", path)
        m = YOLO(path)
        if _validate_fire_model(m, path):
            return m, path
    except Exception as exc:
        logger.warning("[FIRE] Failed to load %s: %s", path, exc)
    return None


def _try_load_huggingface() -> Optional[Tuple[YOLO, str]]:
    """
    Download and load a community fire/smoke YOLO from HuggingFace.

    Default repo: arnabdhar/YOLOv8-Fire-and-Smoke-Detection (D-Fire-trained
    YOLOv8n, ~6 MB). Override with FIRE_HF_REPO / FIRE_HF_FILENAME if you
    prefer a different model.

    Returns None on any failure (missing dependency, no network, repo gone),
    so the caller can fall through to the YOLO-World fallback.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "[FIRE] huggingface_hub not installed — skipping HF fire model. "
            "Run: pip install huggingface_hub"
        )
        return None

    try:
        logger.info("[FIRE] Fetching HF fire model: %s/%s", _HF_REPO, _HF_FILENAME)
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
        logger.info("[FIRE] HF weights cached at: %s", weights_path)
        m = YOLO(weights_path)
        source = f"hf:{_HF_REPO}/{_HF_FILENAME}"
        if _validate_fire_model(m, source):
            return m, source
    except Exception as exc:
        logger.warning("[FIRE] HuggingFace download/load failed: %s", exc)
    return None


def _load_world_fallback() -> Tuple[YOLO, str]:
    """Load YOLO-World prompted with fire vocabulary (last-resort fallback)."""
    logger.info("[FIRE] Loading YOLO-World fallback: %s", _WORLD_MODEL_PATH)
    m = YOLO(_WORLD_MODEL_PATH)
    m.set_classes(_WORLD_FIRE_PROMPTS)
    logger.info("[FIRE] YOLO-World prompts: %s", _WORLD_FIRE_PROMPTS)
    return m, _WORLD_MODEL_PATH


def _get_model() -> YOLO:
    """Return the cached YOLO fire model, loading it on first call."""
    global _model, _model_kind, _model_source
    if _model is not None:
        return _model

    # Priority: custom env var → HuggingFace D-Fire YOLO → YOLO-World.
    custom = _try_load_custom()
    if custom is not None:
        _model, _model_source = custom
        _model_kind = "custom"
    else:
        hf = _try_load_huggingface()
        if hf is not None:
            _model, _model_source = hf
            _model_kind = "huggingface"
        else:
            _model, _model_source = _load_world_fallback()
            _model_kind = "world"

    logger.info("[FIRE] Active detector: kind=%s source=%s device=%s hsv_verify=%s",
                _model_kind, _model_source, _DEVICE, _USE_HSV_VERIFICATION)
    return _model


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_C_RED   = (0,   0, 255)
_C_WHITE = (255, 255, 255)
_C_GREEN = (0, 210,   0)


def _scale(frame: "cv2.Mat") -> float:
    """Per-frame UI scale factor relative to a 720p reference height."""
    return max(1.0, frame.shape[0] / 720.0)


def _draw_box(
    frame: "cv2.Mat",
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw a filled-label bounding box, scaled to frame size."""
    s          = _scale(frame)
    thickness  = max(2, int(round(3 * s)))
    font_scale = 0.70 * s
    text_thick = max(1, int(round(2 * s)))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
    pad = int(round(6 * s))
    label_y = max(y1 - pad, lh + pad)
    cv2.rectangle(frame,
                  (x1, label_y - lh - pad),
                  (x1 + lw + 2 * pad, label_y + 2),
                  color, -1)
    cv2.putText(frame, label, (x1 + pad, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _C_WHITE,
                text_thick, cv2.LINE_AA)


def _draw_fire_banner(frame: "cv2.Mat") -> None:
    h, w = frame.shape[:2]
    s          = _scale(frame)
    bar_h      = int(round(50 * s))
    font_scale = 0.85 * s
    text_thick = max(2, int(round(2 * s)))

    cv2.rectangle(frame, (0, 0), (w, bar_h), _C_RED, -1)
    cv2.line(frame, (0, bar_h), (w, bar_h), _C_WHITE, max(1, int(round(s))))
    cv2.putText(frame, "  FIRE DETECTED -- EVACUATE IMMEDIATELY",
                (int(round(10 * s)), int(round(35 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                _C_WHITE, text_thick, cv2.LINE_AA)


def _draw_safe_banner(frame: "cv2.Mat") -> None:
    h, w = frame.shape[:2]
    s          = _scale(frame)
    bar_h      = int(round(32 * s))
    font_scale = 0.60 * s
    text_thick = max(1, int(round(2 * s)))

    cv2.rectangle(frame, (0, 0), (w, bar_h), _C_GREEN, -1)
    cv2.line(frame, (0, bar_h), (w, bar_h), _C_WHITE, max(1, int(round(s))))
    cv2.putText(frame, "  NO FIRE DETECTED",
                (int(round(10 * s)), int(round(22 * s))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                _C_WHITE, text_thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Detection engine
# ---------------------------------------------------------------------------

def _is_fire_label(label: str) -> bool:
    """Whether a model class label is fire-related (case-insensitive)."""
    lab = label.lower()
    return any(kw in lab for kw in _FIRE_LABEL_KEYWORDS)


def _has_fire_colors(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[bool, float]:
    """
    HSV verification: does the bbox region contain enough fire-colored pixels?

    Returns (passes, ratio). The ratio is the fraction of pixels in the box
    that fall inside saturated red-orange-yellow HSV bands.
    """
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return False, 0.0

    roi = frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return False, 0.0

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, _HSV_LOW_1, _HSV_HIGH_1)
    m2   = cv2.inRange(hsv, _HSV_LOW_2, _HSV_HIGH_2)
    mask = cv2.bitwise_or(m1, m2)

    fire_px = int(np.count_nonzero(mask))
    total   = int(mask.size)
    ratio   = fire_px / total if total > 0 else 0.0
    return (ratio >= _MIN_FIRE_PIXEL_RATIO), ratio


def _yolo_detect(model: YOLO, frame: np.ndarray, frame_idx: int) -> bool:
    """
    Run YOLO inference and draw confirmed boxes.

    HSV color verification is OFF by default (the dedicated D-Fire model
    is reliable enough that an extra heuristic gate would only suppress
    real detections). Set FIRE_HSV_VERIFY=1 to re-enable it as a
    secondary filter — useful only when the YOLO-World fallback is active
    in noisy lighting.
    """
    try:
        # imgsz handles letterboxing internally — we pass the original frame
        # so aspect ratio is preserved (portrait videos no longer get squished).
        results = model(
            frame,
            conf    = _YOLO_CONF,
            iou     = _IOU_THRESH,
            imgsz   = _INFER_IMGSZ,
            device  = _DEVICE,
            verbose = False,
        )
    except Exception as exc:
        logger.warning("[FIRE] YOLO error on frame %d: %s", frame_idx, exc)
        return False

    fire_found  = False
    raw_count   = 0
    for r in results:
        if r.boxes is None:
            continue
        names = r.names if r.names is not None else model.names
        for box in r.boxes:
            raw_count += 1
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = names[cls] if cls in names else str(cls)

            is_fire = _is_fire_label(label)
            passes_conf = conf >= _ACCEPT_CONF

            if _VERBOSE_DETECTIONS:
                tag = (
                    "HIT"      if (is_fire and passes_conf) else
                    "LOW_CONF" if (is_fire and not passes_conf) else
                    "OTHER"
                )
                logger.info(
                    "[FIRE] frame=%d  %s  label=%r conf=%.3f",
                    frame_idx, tag, label, conf,
                )

            if not is_fire:
                continue
            if not passes_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Optional HSV gate (off by default).
            ratio = -1.0
            if _USE_HSV_VERIFICATION:
                ok, ratio = _has_fire_colors(frame, x1, y1, x2, y2)
                if not ok:
                    logger.info(
                        "[FIRE] frame=%d  HSV_REJECT  conf=%.2f ratio=%.3f label=%s",
                        frame_idx, conf, ratio, label,
                    )
                    continue

            fire_found = True
            box_text = (
                f"FIRE {conf:.2f}"
                if ratio < 0 else
                f"FIRE {conf:.2f} ({ratio*100:.0f}%)"
            )
            _draw_box(frame, x1, y1, x2, y2, box_text, _C_RED)

    if _VERBOSE_DETECTIONS and raw_count == 0:
        logger.info("[FIRE] frame=%d  EMPTY  (model produced 0 raw detections)", frame_idx)
    return fire_found

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_fire_video(
    video_path:        str,
    output_video_path: Optional[str] = None,
    draw_banner:       bool          = True,
) -> Dict[str, Any]:
    """
    Analyse a video for fire hazards using YOLO.

    Parameters
    ----------
    video_path        : path to input video.
    output_video_path : if given, an annotated video is written here.
    draw_banner       : when True (default), the FIRE DETECTED / NO FIRE
                        DETECTED banner is drawn at the top of every frame.
                        Set to False when chaining fire after another module
                        that already drew its own top banner -- the fire
                        bbox+label still gets drawn, but the banner is
                        suppressed so we don't paint over the upstream UI.
    """
    model    = _get_model()
    out_path = output_video_path or _OUTPUT_FILE

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_video(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS)            or 25.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pick output dimensions: keep the source aspect ratio, optionally
    # capped by FIRE_OUTPUT_MAX_SIDE so we don't write a 100 MB file for
    # every 4K clip. _OUTPUT_MAX_SIDE = 0 means "no cap, write at source".
    if _OUTPUT_MAX_SIDE > 0 and max(src_w, src_h) > _OUTPUT_MAX_SIDE:
        scale = _OUTPUT_MAX_SIDE / float(max(src_w, src_h))
        out_w = int(round(src_w * scale))
        out_h = int(round(src_h * scale))
    else:
        out_w, out_h = src_w, src_h

    logger.info(
        "[FIRE] Source %dx%d @ %.1f fps — inference imgsz=%d, output %dx%d",
        src_w, src_h, fps, _INFER_IMGSZ, out_w, out_h,
    )

    writer: Optional[cv2.VideoWriter] = None
    total_frames = 0
    fire_frames  = 0
    raw_idx      = 0

    # Temporal persistence: a frame counts as fire-positive only after the
    # raw detector has fired on _FIRE_PERSISTENCE_FRAMES consecutive frames.
    fire_streak  = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if raw_idx % _FRAME_STRIDE != 0:
                continue

            total_frames += 1

            # ── YOLO Detection ──────────────────────────────────────────
            # Pass the ORIGINAL frame to the model — ultralytics
            # letterboxes to imgsz internally and returns boxes in the
            # original frame coordinates, so annotations land correctly.
            raw_hit = _yolo_detect(model, frame, raw_idx)

            if raw_hit:
                fire_streak += 1
            else:
                fire_streak = 0

            confirmed = fire_streak >= _FIRE_PERSISTENCE_FRAMES

            if confirmed:
                fire_frames += 1
                if draw_banner:
                    _draw_fire_banner(frame)
                logger.info(
                    "[FIRE] Frame %d — fire confirmed (streak=%d)",
                    raw_idx, fire_streak,
                )
            else:
                if draw_banner:
                    _draw_safe_banner(frame)

            # Resize for output only — never for inference.
            if (out_w, out_h) != (frame.shape[1], frame.shape[0]):
                out_frame = cv2.resize(frame, (out_w, out_h),
                                       interpolation=cv2.INTER_AREA)
            else:
                out_frame = frame

            # ── Init writer ───────────────────────────────────────────────
            if writer is None:
                try:
                    effective_fps = max(1.0, fps / _FRAME_STRIDE)
                    writer = get_video_writer(
                        out_path,
                        fps    = effective_fps,
                        width  = out_w,
                        height = out_h,
                    )
                except RuntimeError as exc:
                    logger.error("[FIRE] Cannot create VideoWriter: %s", exc)
                    break

            writer.write(out_frame)

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
        "[FIRE] Done — detector=%s(%s) | %d frames | fire_frames=%d | ratio=%.3f | status=%s",
        _model_kind, _model_source,
        total_frames, fire_frames, fire_ratio, status,
    )
    print(f"[FIRE SUMMARY] detector={_model_kind}({_model_source}) "
          f"| total={total_frames} | fire_frames={fire_frames} "
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
