"""
ppe_utils.py
============
Utility helpers for the PPE Detection module.

Handles:
  - Label normalisation for your trained model classes
    (helmet, vest, gloves, boots, human)
  - Bounding-box tensor → Python list
  - Per-frame SAFE/UNSAFE logic
  - Video-level compliance aggregation
  - Annotated frame drawing with color-coded boxes
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1.  Label normalisation
#     Maps any variant a YOLO model might output → canonical name
# ---------------------------------------------------------------------------

_LABEL_MAP: Dict[str, str] = {
    # Human / person
    "human":        "human",
    "person":       "human",
    # Helmet / hard hat
    "helmet":       "helmet",
    "hard hat":     "helmet",
    "hardhat":      "helmet",
    "hard_hat":     "helmet",
    # Safety vest
    "vest":         "vest",
    "safety vest":  "vest",
    "safety_vest":  "vest",
    "hi-vis vest":  "vest",
    # Gloves
    "gloves":       "gloves",
    "glove":        "gloves",
    "safety gloves":"gloves",
    # Boots / safety shoes
    "boots":        "boots",
    "boot":         "boots",
    "safety boots": "boots",
    "safety shoes": "boots",
}

# PPE items that count toward compliance (NOT "human")
PPE_ITEMS = {"helmet", "vest", "gloves", "boots"}

# Minimum PPE required to be SAFE (helmet + vest are mandatory)
REQUIRED_PPE = {"helmet", "vest"}


def normalize_label(raw: str) -> str:
    """Return canonical label string; falls back to lowercased raw label."""
    return _LABEL_MAP.get(raw.lower().strip(), raw.lower().strip())


# ---------------------------------------------------------------------------
# 2.  Bounding box helper
# ---------------------------------------------------------------------------

def bbox_to_list(box_xyxy) -> List[float]:
    """Convert tensor/array [x1,y1,x2,y2] → plain Python list, 2 dp."""
    return [round(float(v), 2) for v in box_xyxy]


# ---------------------------------------------------------------------------
# 2b.  PPE assignment — center-point containment
# ---------------------------------------------------------------------------

# Why center-point and NOT IoU:
#   A helmet bbox covers ~5% of a person bbox area.
#   IoU(person, helmet) ≈ 0.04 — always below any reasonable threshold.
#   The correct question is: "Is the PPE item centered inside the person box?"
#   A helmet on a person's head will always have its center within the person's
#   bounding box, so center-point containment is the reliable test.

_MIN_DET_CONF: float = 0.50   # reject detections below this confidence


def is_ppe_on_person(person_box: List[float], item_box: List[float]) -> bool:
    """
    Return True when the CENTER of item_box lies inside person_box.

    This is intentionally lenient: we only need the PPE item's midpoint
    to fall within the person rectangle.  This correctly handles helmets
    (small, near the top of the person box) and vests (larger, but still
    centred on the torso which is always inside the person box).

    Parameters
    ----------
    person_box : [x1, y1, x2, y2]
    item_box   : [x1, y1, x2, y2]
    """
    px1, py1, px2, py2 = person_box
    ix1, iy1, ix2, iy2 = item_box
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    return (px1 < cx < px2) and (py1 < cy < py2)


def evaluate_frame_safety(
    detections: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Decide per-person PPE compliance using center-point containment.

    A PPE item is considered "worn" by a person when:
      1. Its detection confidence >= _MIN_DET_CONF (0.50)
      2. The CENTER of its bounding box lies INSIDE the person's bounding box

    If NO person is visible in the frame → safe (nothing to check).
    If ANY person is missing required PPE  → unsafe.

    Returns
    -------
    (is_safe, missing_items)
    """
    # ── 1. Drop low-confidence detections ────────────────────────────────────
    dets = [d for d in detections if d["confidence"] >= _MIN_DET_CONF]

    has_human = any(d["label"] == "human" for d in dets)
    if not has_human:
        # No person visible → treat as safe (no one to check)
        return True, []

    # ── 2. Separate person boxes and PPE boxes ────────────────────────────────
    person_boxes = [d["bbox"] for d in dets if d["label"] == "human"]

    helmet_boxes = [d["bbox"] for d in dets if d["label"] == "helmet"]
    vest_boxes   = [d["bbox"] for d in dets if d["label"] == "vest"]

    # Debug info
    import logging as _log
    _logger = _log.getLogger(__name__)
    _logger.debug(
        "[PPE] persons=%d  helmets=%d  vests=%d",
        len(person_boxes), len(helmet_boxes), len(vest_boxes),
    )

    # ── 3. Per-person assignment via center-point containment ─────────────────
    missing_set: set = set()

    for person_box in person_boxes:
        helmet_on = any(is_ppe_on_person(person_box, hb) for hb in helmet_boxes)
        vest_on   = any(is_ppe_on_person(person_box, vb) for vb in vest_boxes)

        _logger.debug(
            "[PPE] person %s → helmet=%s  vest=%s",
            [int(v) for v in person_box], helmet_on, vest_on,
        )

        if not helmet_on:
            missing_set.add("helmet")
        if not vest_on:
            missing_set.add("vest")

    missing  = sorted(missing_set)
    is_safe  = len(missing) == 0
    return is_safe, missing


# ---------------------------------------------------------------------------
# 4.  Video-level compliance summary
# ---------------------------------------------------------------------------

def compute_summary(
    frame_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate per-frame results into the final compliance summary.

    Parameters
    ----------
    frame_results : list of dict
        Each dict: {frame_id, safe, missing, detections}

    Returns
    -------
    dict with keys matching the required output JSON schema.
    """
    total   = len(frame_results)
    if total == 0:
        return {
            "total_frames":     0,
            "safe_frames":      0,
            "unsafe_frames":    0,
            "compliance_score": 0.0,
            "violations":       [],
        }

    safe_frames   = sum(1 for f in frame_results if f["safe"])
    unsafe_frames = total - safe_frames
    compliance    = round((safe_frames / total) * 100, 2)

    violations = [
        {"frame": f["frame_id"], "missing": f["missing"]}
        for f in frame_results
        if not f["safe"]
    ]

    return {
        "total_frames":     total,
        "safe_frames":      safe_frames,
        "unsafe_frames":    unsafe_frames,
        "compliance_score": compliance,
        "violations":       violations,
    }


# ---------------------------------------------------------------------------
# 5.  Annotation / Drawing
# ---------------------------------------------------------------------------

# BGR colour palette (OpenCV = BGR not RGB)
_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "human":  (255, 200,  50),   # light blue
    "helmet": (  0, 210,   0),   # green
    "vest":   (  0, 165, 255),   # orange
    "gloves": (255,   0, 200),   # magenta
    "boots":  (  0, 220, 220),   # yellow
}
_MISSING_COLOUR: Tuple[int, int, int] = (0, 0, 255)   # red for unknown/alerts


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    is_safe: bool,
    missing: List[str],
) -> np.ndarray:
    """
    Draw colour-coded bounding boxes and a safety status banner on *frame*.

    Returns a new annotated copy (does not modify original).
    """
    out = frame.copy()

    for det in detections:
        label = det["label"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        colour = _COLOURS.get(label, _MISSING_COLOUR)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(
            out, text, (x1 + 1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Top-left safety banner
    banner_colour = (0, 200, 0) if is_safe else (0, 0, 230)
    banner_text   = "SAFE" if is_safe else f"UNSAFE – missing: {', '.join(missing)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), banner_colour, -1)
    cv2.putText(
        out, banner_text, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return out
