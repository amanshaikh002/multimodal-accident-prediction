"""
pose_utils.py
=============
Pose ergonomics utility functions for the backend service layer.

This module is a backend-local copy of the shared utils.py that was
used by the original Streamlit app (app_yolo.py). It provides:

  - COCO-17 keypoint index map (KP)
  - Skeleton pair list (SKELETON)
  - Label / colour mappings
  - Vector-based angle calculations
  - Primary-person selection
  - Keypoint normalisation
  - SmoothingBuffer (temporal angle smoothing)
  - extract_all_features() — one-stop feature extraction
  - build_feature_vector() — assembles numpy array for classifier
  - Reason / violation string builder
  - Frame annotation helpers (skeleton draw, HUD overlay)

Usage inside the backend
------------------------
    from utils.pose_utils import (
        FEATURE_COLS, LABEL_NAMES, make_empty_buffers,
        extract_all_features, build_feature_vector,
        select_primary_person, is_pose_valid,
        build_violation_reason, draw_skeleton, draw_hud_overlay,
    )
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO-17 Keypoint Index Map
# ---------------------------------------------------------------------------

KP: Dict[str, int] = {
    "nose": 0,
    "left_eye": 1,  "right_eye": 2,
    "left_ear": 3,  "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

# Skeleton bone pairs for drawing
SKELETON: List[Tuple[int, int]] = [
    (5, 6),   (5, 7),   (7, 9),
    (6, 8),   (8, 10),  (5, 11),
    (6, 12),  (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
    (0, 5),   (0, 6),
]

# Label mappings
LABEL_NAMES: Dict[int, str] = {0: "SAFE", 1: "MODERATE", 2: "UNSAFE"}
LABEL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),    # green
    1: (0, 200, 255),  # yellow
    2: (0, 0, 220),    # red
}

# Minimum confidence for a joint to be trusted
MIN_KP_CONF: float = 0.40

# Joints that MUST be visible for a valid pose frame
ESSENTIAL_JOINTS: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
]

# ---------------------------------------------------------------------------
# Ergonomic Thresholds (used by rule-based override AND auto_label.py)
# ---------------------------------------------------------------------------

# Back angle (vertex at hip, shoulder→hip→knee; 180°=upright)
BACK_UNSAFE_LOW:   float = 120.0   # < this → always UNSAFE
BACK_MODERATE_LOW: float = 140.0   # < this (+ locked knees) → UNSAFE
BACK_SAFE_MIN:     float = 150.0   # > this + knee in range → SAFE
BACK_UPRIGHT:      float = 160.0   # > this + knee > 150° → standing SAFE

# Knee angle (vertex at knee, hip→knee→ankle; 180°=extended)
KNEE_LOCKED:       float = 150.0   # > this with bent back → UNSAFE
KNEE_SQUAT_MIN:    float = 70.0    # lower bound for proper lifting
KNEE_SQUAT_MAX:    float = 120.0   # upper bound for proper lifting

# Neck angle (vertex at shoulder; 180°=aligned)
NECK_UNSAFE_LOW:   float = 120.0   # < this → UNSAFE

# ML probability thresholds (Part 7 of spec)
PROB_UNSAFE_THRESH: float = 0.70   # prob_unsafe > this → UNSAFE
PROB_SAFE_THRESH:   float = 0.60   # prob_safe   > this → SAFE

# ---------------------------------------------------------------------------
# Canonical feature column order (must match training)
# ---------------------------------------------------------------------------

# Canonical feature column order (must match training)
FEATURE_COLS: List[str] = [
    "back_angle", "knee_angle", "neck_angle",
    "norm_shoulder_x", "norm_shoulder_y",
    "norm_hip_x",      "norm_hip_y",
    "norm_knee_x",     "norm_knee_y",
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vec2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _magnitude(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _angle_from_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    m1, m2 = _magnitude(v1), _magnitude(v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return float("nan")
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, dot / (m1 * m2))))))


def _angle_at_vertex(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    return _angle_from_vectors(_vec2(b, a), _vec2(b, c))


def _midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


# ---------------------------------------------------------------------------
# Ergonomic angles
# ---------------------------------------------------------------------------

def calc_back_angle(shoulder: Tuple[float, float], hip: Tuple[float, float], knee: Tuple[float, float]) -> float:
    return _angle_at_vertex(shoulder, hip, knee)


def knee_angle_flex(hip, knee, ankle) -> float:
    return _angle_at_vertex(hip, knee, ankle)


def neck_angle_tilt(head, shoulder, hip) -> float:
    return _angle_at_vertex(head, shoulder, hip)


def elbow_angle_flex(shoulder, elbow, wrist) -> float:
    return _angle_at_vertex(shoulder, elbow, wrist)


# ---------------------------------------------------------------------------
# Primary-person selection
# ---------------------------------------------------------------------------

def select_primary_person(
    boxes_xyxy: np.ndarray,
    frame_w: int = 640,
    frame_h: int = 480,
) -> Optional[int]:
    """Return index of largest / most-central person in frame."""
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    cx_f, cy_f = frame_w / 2.0, frame_h / 2.0
    best_idx, best_score = None, None
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        dist = math.hypot((cx - cx_f) / frame_w, (cy - cy_f) / frame_h)
        score = (-area, dist)
        if best_score is None or score < best_score:
            best_score, best_idx = score, i
    return best_idx


# ---------------------------------------------------------------------------
# Pose validation
# ---------------------------------------------------------------------------

def is_pose_valid(kps_conf: np.ndarray) -> bool:
    for name in ESSENTIAL_JOINTS:
        if float(kps_conf[KP[name]]) < MIN_KP_CONF:
            return False
    return True


# ---------------------------------------------------------------------------
# Raw joint extraction
# ---------------------------------------------------------------------------

def extract_raw_joints(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    def pt(name: str) -> Tuple[float, float]:
        idx = KP[name]
        return (float(kps_xy[idx][0]), float(kps_xy[idx][1]))

    left_conf  = sum(float(kps_conf[KP[k]]) for k in
                     ["left_shoulder", "left_elbow", "left_wrist",
                      "left_hip", "left_knee", "left_ankle"])
    right_conf = sum(float(kps_conf[KP[k]]) for k in
                     ["right_shoulder", "right_elbow", "right_wrist",
                      "right_hip", "right_knee", "right_ankle"])
    use_left = left_conf >= right_conf

    l_sh, r_sh = pt("left_shoulder"),  pt("right_shoulder")
    l_hp, r_hp = pt("left_hip"),       pt("right_hip")
    l_kn, r_kn = pt("left_knee"),      pt("right_knee")
    l_el, r_el = pt("left_elbow"),     pt("right_elbow")
    l_wr, r_wr = pt("left_wrist"),     pt("right_wrist")
    l_an, r_an = pt("left_ankle"),     pt("right_ankle")
    l_ea, r_ea = pt("left_ear"),       pt("right_ear")
    nose       = pt("nose")

    ear_conf = float(kps_conf[KP["left_ear"]]) + float(kps_conf[KP["right_ear"]])
    head = _midpoint(l_ea, r_ea) if ear_conf > 0.5 else nose

    return {
        "shoulder_mid":  _midpoint(l_sh, r_sh),
        "hip_mid":       _midpoint(l_hp, r_hp),
        "knee_mid":      _midpoint(l_kn, r_kn),
        "elbow_mid":     _midpoint(l_el, r_el),
        "head":          head,
        "shoulder_side": l_sh if use_left else r_sh,
        "elbow_side":    l_el if use_left else r_el,
        "wrist_side":    l_wr if use_left else r_wr,
        "hip_side":      l_hp if use_left else r_hp,
        "knee_side":     l_kn if use_left else r_kn,
        "ankle_side":    l_an if use_left else r_an,
    }


# ---------------------------------------------------------------------------
# Keypoint normalisation
# ---------------------------------------------------------------------------

def normalize_keypoints(joints: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    origin = joints["hip_mid"]
    torso_vec = _vec2(joints["hip_mid"], joints["shoulder_mid"])
    torso_len = _magnitude(torso_vec)
    if torso_len < 1e-3:
        return {k: (0.0, 0.0) for k in joints}
    return {
        name: ((x - origin[0]) / torso_len, (y - origin[1]) / torso_len)
        for name, (x, y) in joints.items()
    }


# ---------------------------------------------------------------------------
# Temporal smoothing buffer
# ---------------------------------------------------------------------------

class SmoothingBuffer:
    """Rolling-window moving-average + velocity + acceleration."""

    def __init__(self, window: int = 8) -> None:
        self._buf: deque = deque(maxlen=window)

    def push(self, value: float) -> None:
        if not math.isnan(value):
            self._buf.append(value)

    def mean(self) -> float:
        return float(np.mean(list(self._buf))) if self._buf else float("nan")

    def velocity(self) -> float:
        return float(self._buf[-1] - self._buf[-2]) if len(self._buf) >= 2 else float("nan")

    def acceleration(self) -> float:
        if len(self._buf) < 3:
            return float("nan")
        return float((self._buf[-1] - self._buf[-2]) - (self._buf[-2] - self._buf[-3]))

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


def make_empty_buffers() -> Dict[str, SmoothingBuffer]:
    return {
        "back":  SmoothingBuffer(8),
        "knee":  SmoothingBuffer(8),
        "neck":  SmoothingBuffer(8),
        "elbow": SmoothingBuffer(8),
    }


# ---------------------------------------------------------------------------
# Full feature extraction (training & inference shared)
# ---------------------------------------------------------------------------

def extract_all_features(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    buffers: Dict[str, SmoothingBuffer],
) -> Optional[Dict[str, float]]:
    """Extract all 15 ergonomic features from one frame. Returns None if invalid."""
    if not is_pose_valid(kps_conf):
        return None

    joints = extract_raw_joints(kps_xy, kps_conf)

    raw_back  = calc_back_angle(joints["shoulder_mid"], joints["hip_mid"], joints["knee_mid"])
    raw_knee  = knee_angle_flex(joints["hip_side"],  joints["knee_side"],  joints["ankle_side"])
    raw_neck  = neck_angle_tilt(joints["head"],       joints["shoulder_mid"], joints["hip_mid"])
    raw_elbow = elbow_angle_flex(joints["shoulder_side"], joints["elbow_side"], joints["wrist_side"])

    if any(math.isnan(a) for a in [raw_back, raw_knee, raw_neck]):
        return None

    for name, val in [("back", raw_back), ("knee", raw_knee),
                      ("neck", raw_neck), ("elbow", raw_elbow)]:
        buffers[name].push(val)

    def safe(v: float) -> float:
        return 0.0 if math.isnan(v) else v

    norm = normalize_keypoints(joints)

    return {
        "back_angle":       buffers["back"].mean(),
        "knee_angle":       buffers["knee"].mean(),
        "neck_angle":       buffers["neck"].mean(),
        "elbow_angle":      safe(buffers["elbow"].mean()) if not math.isnan(buffers["elbow"].mean()) else 180.0,
        "back_vel":         safe(buffers["back"].velocity()),
        "knee_vel":         safe(buffers["knee"].velocity()),
        "neck_vel":         safe(buffers["neck"].velocity()),
        "back_acc":         safe(buffers["back"].acceleration()),
        "knee_acc":         safe(buffers["knee"].acceleration()),
        "norm_shoulder_x":  norm["shoulder_mid"][0],
        "norm_shoulder_y":  norm["shoulder_mid"][1],
        "norm_hip_x":       norm["hip_mid"][0],
        "norm_hip_y":       norm["hip_mid"][1],
        "norm_knee_x":      norm["knee_mid"][0],
        "norm_knee_y":      norm["knee_mid"][1],
    }


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Assemble the canonical (1, 9) feature vector for classifier input."""
    vec = [features.get(col, 0.0) for col in FEATURE_COLS]
    arr = np.array([vec], dtype=np.float32)
    return arr


# ---------------------------------------------------------------------------
# Violation reason builder
# ---------------------------------------------------------------------------

def build_violation_reason(features: Dict[str, float]) -> str:
    """Return a human-readable string describing the worst ergonomic issue.
    
    Uses VERTEX ANGLE convention: 180° = straight, lower = more bent.
    """
    reasons: List[str] = []
    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    if not math.isnan(ba):
        if ba < BACK_UNSAFE_LOW:
            reasons.append("Excessive back bending")
        elif ba < BACK_MODERATE_LOW:
            reasons.append("Moderate back lean")

    if not math.isnan(ka):
        if ka > KNEE_LOCKED:
            reasons.append("Stiff legs — bend knees")
        elif ka > 140.0:
            reasons.append("Moderate knee stiffness")

    if not math.isnan(na):
        if na < NECK_UNSAFE_LOW:
            reasons.append("Significant neck tilt")
        elif na < 150.0:
            reasons.append("Slight neck forward")

    return " | ".join(reasons) if reasons else "Bad lifting posture"

# ---------------------------------------------------------------------------
# Hybrid Rule + ML Classification
# ---------------------------------------------------------------------------

def hybrid_classify(
    features: Dict[str, float],
    classifier,
    x_vec: np.ndarray,
) -> Tuple[int, float, str]:
    """
    Hybrid posture classifier combining strict geometric rules (primary)
    and ML probability scores (secondary refinement).

    Decision hierarchy:
      1. Extreme cases are handled by geometric rule-based overrides FIRST.
      2. For ambiguous cases, ML probability thresholds decide.
      3. Default fallback: MODERATE.

    Returns:
        (label_id, confidence, decision_source)
        label_id:         0=SAFE, 1=MODERATE, 2=UNSAFE
        confidence:       float [0, 1]
        decision_source:  'rule' or 'ml'
    """
    ba = features.get("back_angle",  180.0)
    ka = features.get("knee_angle",  180.0)
    na = features.get("neck_angle",  180.0)

    # ── GEOMETRIC OVERRIDES (non-negotiable) ───────────────────────────────

    # 1. Extreme forward bend → UNSAFE (rule is always right here)
    if ba < BACK_UNSAFE_LOW:
        return 2, 0.92, "rule"

    # 2. Bent back + locked knees → UNSAFE (classic bad lifting)
    if ba < BACK_MODERATE_LOW and ka > KNEE_LOCKED:
        return 2, 0.88, "rule"

    # 3. Severe neck forward head → UNSAFE
    if na < NECK_UNSAFE_LOW:
        return 2, 0.85, "rule"

    # 4. Proper lifting posture (straight back + bent knees) → SAFE
    if ba > BACK_SAFE_MIN and KNEE_SQUAT_MIN <= ka <= KNEE_SQUAT_MAX:
        return 0, 0.90, "rule"

    # 5. Standing upright → SAFE
    if ba > BACK_UPRIGHT and ka > KNEE_LOCKED:
        return 0, 0.92, "rule"

    # ── ML PROBABILITY-BASED DECISION (ambiguous zone) ─────────────────────
    try:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(x_vec)[0]
            n_cls = len(proba)
            prob_safe     = float(proba[0]) if n_cls > 0 else 0.0
            prob_moderate = float(proba[1]) if n_cls > 1 else 0.0
            prob_unsafe   = float(proba[2]) if n_cls > 2 else 0.0

            if prob_unsafe > PROB_UNSAFE_THRESH:
                return 2, prob_unsafe, "ml"
            elif prob_safe > PROB_SAFE_THRESH:
                return 0, prob_safe, "ml"
            else:
                # Argmax as tiebreaker but cap at MODERATE
                conf = max(prob_safe, prob_moderate, prob_unsafe)
                return 1, conf, "ml"
        else:
            raw = int(classifier.predict(x_vec)[0])
            return raw, 0.75, "ml"
    except Exception:
        # If ML fails for any reason → default MODERATE (safe fallback)
        return 1, 0.50, "rule"


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def draw_skeleton(
    frame: np.ndarray,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_thr: float = 0.30,
) -> None:
    """Draw COCO-17 skeleton bones and joint dots on the frame (in-place)."""
    for a, b in SKELETON:
        if float(kps_conf[a]) >= conf_thr and float(kps_conf[b]) >= conf_thr:
            p1 = (int(kps_xy[a][0]), int(kps_xy[a][1]))
            p2 = (int(kps_xy[b][0]), int(kps_xy[b][1]))
            cv2.line(frame, p1, p2, (255, 230, 50), 2, cv2.LINE_AA)
    for i in range(len(kps_xy)):
        if float(kps_conf[i]) >= conf_thr:
            cv2.circle(
                frame,
                (int(kps_xy[i][0]), int(kps_xy[i][1])),
                4, (0, 255, 255), -1, cv2.LINE_AA,
            )


def draw_hud_overlay(
    frame: np.ndarray,
    features: Dict[str, float],
    label: str,
    confidence: float,
    box_xyxy: np.ndarray,
    color_bgr: Tuple[int, int, int],
    frame_h: int = 480,
) -> None:
    """
    Render bounding box, label badge, and angle HUD on the frame (in-place).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1 = int(box_xyxy[0]), int(box_xyxy[1])
    x2, y2 = int(box_xyxy[2]), int(box_xyxy[3])

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

    # Label badge
    badge = f"{label}  {confidence * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(badge, font, 0.65, 2)
    by = max(y1 - 10, th + 8)
    cv2.rectangle(frame, (x1, by - th - 6), (x1 + tw + 10, by + 4), color_bgr, -1)
    cv2.putText(frame, badge, (x1 + 5, by - 2), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Angle HUD (top-left, semi-transparent)
    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    hud = [
        f"Back : {ba:5.1f}deg" if not math.isnan(ba) else "Back : N/A",
        f"Knee : {ka:5.1f}deg" if not math.isnan(ka) else "Knee : N/A",
        f"Neck : {na:5.1f}deg" if not math.isnan(na) else "Neck : N/A",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (200, 28 + len(hud) * 20 + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (12, 28 + i * 20), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
