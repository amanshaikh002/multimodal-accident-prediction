"""
ppe_utils.py
============
Utility helpers for the PPE Detection module.

Handles:
  - Label normalisation for your trained model classes
    (helmet, vest, gloves, boots, human)
  - Bounding-box tensor → Python list
  - Per-frame SAFE/UNSAFE/UNKNOWN tri-state logic
  - Motion-based occlusion detection
  - Video-level compliance aggregation
  - Annotated frame drawing with color-coded boxes
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Tri-state safety status. UNKNOWN means "we can't see a person but the scene
# is not empty" — a person may be occluded behind machinery, debris, etc.
STATUS_SAFE    = "SAFE"
STATUS_UNSAFE  = "UNSAFE"
STATUS_UNKNOWN = "UNKNOWN"

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

# A helmet bbox covers ~5% of a person bbox area.
#   IoU(person, helmet) ≈ 0.04 — always below any reasonable threshold.
#   The correct question is: "Is the PPE item centered inside the person box?"
#   A helmet on a person's head will always have its center within the person's
#   bounding box, so center-point containment is the reliable test.


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


# ---------------------------------------------------------------------------
# 2c.  Motion score — fast frame differencing for occlusion detection
# ---------------------------------------------------------------------------

# Default motion threshold: fraction of frame pixels that must change
# (after blur + binary threshold) for the scene to count as "active".
MOTION_THRESHOLD_DEFAULT: float = 0.015   # 1.5 % of pixels


def compute_motion_score(
    prev_gray: Optional[np.ndarray],
    curr_gray: Optional[np.ndarray],
) -> float:
    """
    Return the fraction of pixels that changed between two grayscale frames.

    Intended for static cameras: high score with no person detected ⇒ likely
    occlusion or a falling object, not an empty scene.

    Returns 0.0 if either frame is missing or shapes differ.
    """
    if prev_gray is None or curr_gray is None:
        return 0.0
    if prev_gray.shape != curr_gray.shape:
        return 0.0

    diff = cv2.absdiff(prev_gray, curr_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(thresh)) / float(thresh.size)


def apply_hazard_override(
    status: str,
    missing: List[str],
    reason: Optional[str],
    hazard_detections: List[Dict[str, Any]],
) -> Tuple[str, List[str], Optional[str], List[str]]:
    """
    Fold open-vocabulary hazard detections (debris, falling rock, fallen
    person, …) into the tri-state PPE result.

    Hazards are highest-priority — a worker in full PPE next to falling
    debris is still UNSAFE, and a hazard in an empty / occluded scene
    flips UNKNOWN to UNSAFE because the danger is real either way.

    Returns
    -------
    (status, missing, reason, hazard_labels)
        hazard_labels — sorted, deduplicated list of hazard classes seen
                        this frame. Empty when no hazard present.
    """
    if not hazard_detections:
        return status, missing, reason, []

    hazard_labels = sorted({d["label"] for d in hazard_detections})

    if status == STATUS_UNSAFE and reason == "missing_ppe":
        new_reason = "missing_ppe_and_hazard"
    else:
        new_reason = "hazard_detected"

    # Status is UNSAFE regardless of prior PPE/occlusion state.
    return STATUS_UNSAFE, missing, new_reason, hazard_labels


def evaluate_frame_safety(
    detections: List[Dict[str, Any]],
    motion_score: float = 0.0,
    person_recently_seen: bool = False,
    motion_threshold: float = MOTION_THRESHOLD_DEFAULT,
) -> Tuple[str, List[str], Optional[str]]:
    """
    Tri-state PPE compliance decision.

    Status semantics
    ----------------
    SAFE     — at least one person visible AND all required PPE present.
    UNSAFE   — at least one person visible AND some required PPE missing.
    UNKNOWN  — no person visible BUT either the scene is moving (occlusion or
               falling object likely) or a person was seen very recently.
               Refusing to default to SAFE under occlusion is the whole point.

    Only an empty, motionless scene with no recently-seen worker is SAFE
    without a positive person detection.

    Parameters
    ----------
    detections           : filtered detections for this frame
    motion_score         : fraction of changed pixels vs. previous frame
                           (0.0 disables the motion check)
    person_recently_seen : True if a person was detected in the last few frames
    motion_threshold     : motion score above which the scene counts as active

    Returns
    -------
    (status, missing_items, reason)
        status        ∈ {SAFE, UNSAFE, UNKNOWN}
        missing_items list of missing required PPE labels (empty unless UNSAFE)
        reason        short tag describing why (None for SAFE)
    """
    dets = detections

    has_human = any(d["label"] == "human" for d in dets)

    # ── No person detected → tri-state branch, never blindly SAFE ────────────
    if not has_human:
        if motion_score >= motion_threshold:
            return STATUS_UNKNOWN, [], "no_person_but_motion"
        if person_recently_seen:
            return STATUS_UNKNOWN, [], "person_recently_seen"
        return STATUS_SAFE, [], None

    # ── Person(s) detected → per-person PPE check ────────────────────────────
    person_boxes = [d["bbox"] for d in dets if d["label"] == "human"]
    helmet_boxes = [d["bbox"] for d in dets if d["label"] == "helmet"]
    vest_boxes   = [d["bbox"] for d in dets if d["label"] == "vest"]

    import logging as _log
    _logger = _log.getLogger(__name__)
    _logger.debug(
        "[PPE] persons=%d  helmets=%d  vests=%d",
        len(person_boxes), len(helmet_boxes), len(vest_boxes),
    )

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

    missing = sorted(missing_set)
    if missing:
        return STATUS_UNSAFE, missing, "missing_ppe"
    return STATUS_SAFE, [], None


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
        Each dict: {frame_id, status, safe, missing, reason, detections}
        ``safe`` is kept for backward compatibility with downstream services
        but ``status`` is the authoritative tri-state value.

    Returns
    -------
    dict with the per-frame counts (including UNKNOWN) and a flat violations
    list. UNKNOWN frames appear in violations with type ``occlusion`` so the
    frontend / combined pipeline can surface them instead of silently
    dropping into SAFE.
    """
    total = len(frame_results)
    if total == 0:
        return {
            "total_frames":     0,
            "safe_frames":      0,
            "unsafe_frames":    0,
            "unknown_frames":   0,
            "hazard_frames":    0,
            "compliance_score": 0.0,
            "violations":       [],
        }

    safe_frames    = sum(1 for f in frame_results if f.get("status") == STATUS_SAFE)
    unsafe_frames  = sum(1 for f in frame_results if f.get("status") == STATUS_UNSAFE)
    unknown_frames = sum(1 for f in frame_results if f.get("status") == STATUS_UNKNOWN)
    hazard_frames  = sum(1 for f in frame_results if f.get("hazards"))

    compliance = round((safe_frames / total) * 100, 2)

    violations: List[Dict[str, Any]] = []
    for f in frame_results:
        st       = f.get("status")
        hazards  = f.get("hazards") or []
        reason   = f.get("reason")

        if hazards:
            violations.append({
                "frame":   f["frame_id"],
                "type":    "hazard",
                "missing": f.get("missing", []),
                "hazards": hazards,
                "reason":  reason or "hazard_detected",
            })
        elif st == STATUS_UNSAFE:
            violations.append({
                "frame":   f["frame_id"],
                "type":    "missing_ppe",
                "missing": f.get("missing", []),
                "reason":  reason or "missing_ppe",
            })
        elif st == STATUS_UNKNOWN:
            violations.append({
                "frame":   f["frame_id"],
                "type":    "occlusion",
                "missing": [],
                "reason":  reason or "no_person_visible",
            })

    return {
        "total_frames":     total,
        "safe_frames":      safe_frames,
        "unsafe_frames":    unsafe_frames,
        "unknown_frames":   unknown_frames,
        "hazard_frames":    hazard_frames,
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


_UNKNOWN_REASONS: Dict[str, str] = {
    "no_person_but_motion":  "OCCLUDED – motion without visible worker",
    "person_recently_seen":  "OCCLUDED – worker hidden",
    "no_person_visible":     "UNKNOWN – no worker visible",
}

# Bright red for hazard boxes — distinct from PPE missing-item red.
_HAZARD_COLOUR: Tuple[int, int, int] = (40, 40, 240)


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    status: str,
    missing: List[str],
    reason: Optional[str] = None,
    hazard_detections: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Draw colour-coded bounding boxes and a tri-state status banner on *frame*.

    Returns a new annotated copy (does not modify original). When
    ``hazard_detections`` is non-empty, hazard boxes are drawn in bright red
    and the banner names the worst hazard label.
    """
    out = frame.copy()
    hazards = hazard_detections or []

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

    # Hazard boxes drawn last so they overlay everything except the banner.
    for hz in hazards:
        x1, y1, x2, y2 = [int(v) for v in hz["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), _HAZARD_COLOUR, 3)

        text = f"HAZARD: {hz['label']} {hz['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), _HAZARD_COLOUR, -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Top-left safety banner — green/red/yellow for SAFE/UNSAFE/UNKNOWN.
    # Hazards always force a red banner with the hazard label called out.
    if hazards:
        banner_colour = _HAZARD_COLOUR
        hz_labels     = sorted({h["label"] for h in hazards})
        if missing:
            banner_text = f"UNSAFE – HAZARD ({', '.join(hz_labels)}) + missing: {', '.join(missing)}"
        else:
            banner_text = f"UNSAFE – HAZARD: {', '.join(hz_labels)}"
    elif status == STATUS_SAFE:
        banner_colour = (0, 200, 0)
        banner_text   = "SAFE"
    elif status == STATUS_UNSAFE:
        banner_colour = (0, 0, 230)
        banner_text   = f"UNSAFE – missing: {', '.join(missing)}" if missing else "UNSAFE"
    else:  # STATUS_UNKNOWN
        banner_colour = (0, 200, 230)   # amber/yellow in BGR
        banner_text   = _UNKNOWN_REASONS.get(reason or "", "UNKNOWN – worker not visible")

    cv2.rectangle(out, (0, 0), (out.shape[1], 34), banner_colour, -1)
    cv2.putText(
        out, banner_text, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return out
