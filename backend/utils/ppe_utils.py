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
# 3.  Per-frame SAFE / UNSAFE logic
# ---------------------------------------------------------------------------

def evaluate_frame_safety(
    detections: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Given the detections for one frame, decide if any visible person
    is wearing the REQUIRED_PPE items.

    Returns
    -------
    (is_safe, missing_items)
        is_safe      – True if all required PPE detected
        missing_items – list of required labels that were absent
    """
    found_labels = {det["label"] for det in detections}
    has_human    = "human" in found_labels

    # Only evaluate safety when at least one person is in frame
    if not has_human:
        # No person → frame is neutral (treated as safe for compliance)
        return True, []

    missing = [item for item in REQUIRED_PPE if item not in found_labels]
    is_safe = len(missing) == 0
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
