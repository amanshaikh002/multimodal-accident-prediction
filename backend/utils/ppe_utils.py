"""
ppe_utils.py
============
Utility / helper functions for the PPE Detection module.

Responsibilities:
  - Normalize raw YOLO class labels to canonical PPE category names
  - Format bounding-box tensors to plain Python lists
  - Calculate per-video compliance metrics
  - Draw annotated bounding boxes on frames (bonus)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Label Normalization
# ---------------------------------------------------------------------------

# Mapping from any variant a YOLO model might output → canonical PPE label
_LABEL_MAP: Dict[str, str] = {
    # Helmets / hard hats
    "helmet": "helmet",
    "hard hat": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "head protection": "helmet",
    # Safety vests / hi-vis
    "vest": "vest",
    "safety vest": "vest",
    "safety_vest": "vest",
    "hi-vis vest": "vest",
    "hiviz": "vest",
    "high vis": "vest",
    "reflective vest": "vest",
    # Person (kept for reference; not a PPE item)
    "person": "person",
}


def normalize_label(raw_label: str) -> str:
    """
    Convert a raw YOLO detection label to a canonical PPE category.

    Parameters
    ----------
    raw_label : str
        The class name returned by the YOLO model (case-insensitive).

    Returns
    -------
    str
        Canonical label (e.g. ``"helmet"``, ``"vest"``, ``"person"``)
        or the original label lowercased if it is not in the map.
    """
    return _LABEL_MAP.get(raw_label.lower().strip(), raw_label.lower().strip())


def is_ppe_label(label: str) -> bool:
    """Return True if the canonical label is a trackable PPE item."""
    return label in ("helmet", "vest")


# ---------------------------------------------------------------------------
# Bounding Box Helpers
# ---------------------------------------------------------------------------

def bbox_to_list(box_xyxy) -> List[float]:
    """
    Convert an Ultralytics bounding-box tensor/array to a plain Python list
    ``[x1, y1, x2, y2]`` rounded to 2 decimal places.

    Parameters
    ----------
    box_xyxy : torch.Tensor | np.ndarray | list
        A 1-D array-like of length 4 in x1-y1-x2-y2 format.
    """
    return [round(float(v), 2) for v in box_xyxy]


# ---------------------------------------------------------------------------
# Detection Filtering
# ---------------------------------------------------------------------------

def filter_detections(
    raw_detections: List[Dict[str, Any]],
    conf_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Filter raw detections by confidence threshold and normalise labels.

    Parameters
    ----------
    raw_detections : list of dict
        Each dict must have keys: ``label``, ``confidence``, ``bbox``.
    conf_threshold : float
        Minimum confidence to keep a detection (default 0.5).

    Returns
    -------
    list of dict
        Filtered + label-normalised detections.
    """
    filtered = []
    for det in raw_detections:
        if det["confidence"] < conf_threshold:
            continue
        det["label"] = normalize_label(det["label"])
        filtered.append(det)
    return filtered


# ---------------------------------------------------------------------------
# PPE Status per Frame
# ---------------------------------------------------------------------------

def compute_frame_ppe_status(
    detections: List[Dict[str, Any]],
) -> Tuple[bool, bool]:
    """
    Determine whether a helmet and/or vest was detected in a single frame.

    Parameters
    ----------
    detections : list of dict
        Filtered detections with normalised labels.

    Returns
    -------
    (helmet_detected, vest_detected) : tuple of bool
    """
    helmet_detected = False
    vest_detected = False

    for det in detections:
        label = det["label"]
        if label == "helmet":
            helmet_detected = True
        elif label == "vest":
            vest_detected = True

    return helmet_detected, vest_detected


# ---------------------------------------------------------------------------
# Video-level Compliance Metrics
# ---------------------------------------------------------------------------

def compute_compliance_metrics(
    frame_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-frame PPE results into video-level compliance metrics.

    Parameters
    ----------
    frame_results : list of dict
        Each dict must have boolean keys ``helmet`` and ``vest``.

    Returns
    -------
    dict with keys:
        total_frames         : int
        helmet_frames        : int
        vest_frames          : int
        helmet_compliance    : float  (0.0 – 1.0)
        vest_compliance      : float  (0.0 – 1.0)
        status               : str
    """
    total = len(frame_results)
    if total == 0:
        return {
            "total_frames": 0,
            "helmet_frames": 0,
            "vest_frames": 0,
            "helmet_compliance": 0.0,
            "vest_compliance": 0.0,
            "status": "No frames processed",
        }

    helmet_frames = sum(1 for f in frame_results if f.get("helmet", False))
    vest_frames   = sum(1 for f in frame_results if f.get("vest",   False))

    helmet_compliance = round(helmet_frames / total, 4)
    vest_compliance   = round(vest_frames   / total, 4)

    # Determine overall safety status
    if helmet_compliance < 0.5:
        status = "Helmet Missing"
    elif vest_compliance < 0.5:
        status = "Vest Missing"
    else:
        status = "PPE Compliant"

    return {
        "total_frames": total,
        "helmet_frames": helmet_frames,
        "vest_frames": vest_frames,
        "helmet_compliance": helmet_compliance,
        "vest_compliance": vest_compliance,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Annotation (Bonus)
# ---------------------------------------------------------------------------

# Colour palette for known PPE labels (BGR for OpenCV)
_BBOX_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "helmet": (0, 200, 0),    # Green
    "vest":   (0, 165, 255),  # Orange
    "person": (255, 255, 0),  # Cyan
}
_DEFAULT_COLOUR: Tuple[int, int, int] = (0, 0, 255)  # Red for unknowns / missing


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    helmet_detected: bool,
    vest_detected: bool,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of *frame*.

    Boxes are green for known PPE items; red for missing-PPE warning overlay.

    Parameters
    ----------
    frame : np.ndarray
        BGR image array (H × W × 3).
    detections : list of dict
        Filtered detections with keys ``label``, ``confidence``, ``bbox``.
    helmet_detected : bool
    vest_detected   : bool

    Returns
    -------
    np.ndarray
        Annotated frame (same shape as input).
    """
    annotated = frame.copy()

    for det in detections:
        label = det["label"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        colour = _BBOX_COLOURS.get(label, _DEFAULT_COLOUR)

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label text
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), colour, -1)
        cv2.putText(
            annotated, text,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Corner status indicators
    status_lines = [
        ("Helmet: OK" if helmet_detected else "Helmet: MISSING",
         (0, 200, 0) if helmet_detected else (0, 0, 255)),
        ("Vest:   OK" if vest_detected else "Vest:   MISSING",
         (0, 200, 0) if vest_detected else (0, 0, 255)),
    ]
    for i, (txt, col) in enumerate(status_lines):
        cv2.putText(
            annotated, txt,
            (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            col, 2, cv2.LINE_AA,
        )

    return annotated
