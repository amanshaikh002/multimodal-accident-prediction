"""
utils.py — Shared Utility Module
=================================
Worker Pose Safety Monitoring System — YOLOv8-only pipeline.

Provides:
  - COCO-17 keypoint index map
  - Vector-based angle calculations (dot product, no screen-space hack)
  - Body-relative keypoint normalization (hip-centered, torso-scaled)
  - Primary-person selection (largest bbox + closest to frame center)
  - SmoothingBuffer for temporal angle smoothing
  - Temporal feature computation (velocity, acceleration)
  - extract_all_features() — one-stop function for both training & inference
  - build_feature_vector() — assembles the numpy array for the classifier
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# YOLOv8 COCO-17 Keypoint Index Map
# ---------------------------------------------------------------------------
KP: Dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Skeleton pairs for drawing (index pairs from KP)
SKELETON: List[Tuple[int, int]] = [
    (5, 6),   # shoulders
    (5, 7),   # L shoulder → elbow
    (7, 9),   # L elbow → wrist
    (6, 8),   # R shoulder → elbow
    (8, 10),  # R elbow → wrist
    (5, 11),  # L shoulder → hip
    (6, 12),  # R shoulder → hip
    (11, 12), # hips
    (11, 13), # L hip → knee
    (13, 15), # L knee → ankle
    (12, 14), # R hip → knee
    (14, 16), # R knee → ankle
    (0, 5),   # nose → L shoulder
    (0, 6),   # nose → R shoulder
]

# Label mappings
LABEL_NAMES: Dict[int, str] = {0: "SAFE", 1: "MODERATE", 2: "UNSAFE"}
LABEL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 200, 0),    # green — SAFE
    1: (0, 200, 255),  # yellow — MODERATE
    2: (0, 0, 220),    # red — UNSAFE
}

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

# Minimum keypoint confidence to consider a joint reliable
MIN_KP_CONF: float = 0.40

# Essential joints that MUST be visible for a valid pose frame
ESSENTIAL_JOINTS: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]


# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def _vec2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Return vector b − a."""
    return (b[0] - a[0], b[1] - a[1])


def _magnitude(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def angle_from_vectors(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
) -> float:
    """
    Compute the angle (degrees) between two 2-D vectors using the dot product.
    Returns NaN if either vector has zero magnitude.
    Result is clamped to [0, 180].
    """
    mag1 = _magnitude(v1)
    mag2 = _magnitude(v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return float("nan")
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(math.degrees(math.acos(cosine)))


def angle_at_vertex(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """
    Angle (degrees) at point b formed by rays b→a and b→c.
    Uses dot-product formula. Clamped to [0, 180].
    """
    return angle_from_vectors(_vec2(b, a), _vec2(b, c))


def midpoint(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float]:
    """Return the midpoint of two 2-D points."""
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


# ---------------------------------------------------------------------------
# Ergonomic Angle Calculations (relative, NOT screen-space)
# ---------------------------------------------------------------------------

def calc_back_angle(
    shoulder: Tuple[float, float],
    hip: Tuple[float, float],
    knee: Tuple[float, float],
) -> float:
    """
    Angle between the spine vector (shoulder → hip) and the thigh vector (hip → knee).
    """
    return angle_at_vertex(shoulder, hip, knee)


def knee_angle_flex(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
) -> float:
    """
    Knee flexion angle: angle at the knee joint formed by hip→knee→ankle rays.

    Interpretation:
      ~180° = fully extended (standing straight)
      ~90°  = right-angle knee bend (squatting)
    """
    return angle_at_vertex(hip, knee, ankle)


def neck_angle_tilt(
    head: Tuple[float, float],
    shoulder: Tuple[float, float],
    hip: Tuple[float, float],
) -> float:
    """
    Neck / head tilt: angle at the shoulder between head→shoulder direction
    and the torso direction shoulder→hip.

    Interpretation:
      ~180° = head well-aligned with torso
      <130° = significant neck tilt / forward head
    """
    return angle_at_vertex(head, shoulder, hip)


def elbow_angle_flex(
    shoulder: Tuple[float, float],
    elbow: Tuple[float, float],
    wrist: Tuple[float, float],
) -> float:
    """
    Elbow flexion angle at the elbow joint.

    ~180° = fully extended arm
    ~90°  = right-angle elbow bend
    """
    return angle_at_vertex(shoulder, elbow, wrist)


# ---------------------------------------------------------------------------
# Primary Person Selection
# ---------------------------------------------------------------------------

def select_primary_person(
    boxes_xyxy: np.ndarray,
    frame_w: int = 640,
    frame_h: int = 480,
) -> Optional[int]:
    """
    Select the index of the most prominent person from a list of bounding boxes.

    Scoring (lower is better — used for argmin):
      primary key  : -bbox_area  (prefer larger person)
      secondary key: distance_from_center (tie-break by centrality)

    Args:
        boxes_xyxy: shape (N, 4) array of [x1, y1, x2, y2] in pixels
        frame_w, frame_h: frame dimensions for normalization

    Returns:
        Index of the selected person, or None if boxes is empty.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None

    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0

    best_idx: Optional[int] = None
    best_score: Optional[Tuple[float, float]] = None

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        # Normalize center distance (0–1 range)
        dist = math.hypot((cx - cx_frame) / frame_w, (cy - cy_frame) / frame_h)
        score = (-area, dist)  # tuple comparison: largest area first, then closest center
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i

    return best_idx


# ---------------------------------------------------------------------------
# Keypoint Extraction & Normalization
# ---------------------------------------------------------------------------

def is_pose_valid(kps_conf: np.ndarray) -> bool:
    """
    Return True if all essential joints have confidence >= MIN_KP_CONF.
    Rejects frames where the body is partially occluded or poorly detected.
    """
    for name in ESSENTIAL_JOINTS:
        idx = KP[name]
        if float(kps_conf[idx]) < MIN_KP_CONF:
            return False
    return True


def extract_raw_joints(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """
    Build a named-joint dictionary from raw YOLO keypoint arrays.

    Selects the more-visible lateral side (left vs right) for angle computation.
    Uses midpoints for bilateral joints (shoulder, hip, knee).

    Args:
        kps_xy:   shape (17, 2) pixel coordinates
        kps_conf: shape (17,)   per-joint confidence scores

    Returns:
        dict of joint_name → (x, y) in pixels
    """
    def pt(name: str) -> Tuple[float, float]:
        idx = KP[name]
        return (float(kps_xy[idx][0]), float(kps_xy[idx][1]))

    # Determine which lateral side is more visible
    left_conf = sum(float(kps_conf[KP[k]]) for k in [
        "left_shoulder", "left_elbow", "left_wrist",
        "left_hip", "left_knee", "left_ankle",
    ])
    right_conf = sum(float(kps_conf[KP[k]]) for k in [
        "right_shoulder", "right_elbow", "right_wrist",
        "right_hip", "right_knee", "right_ankle",
    ])
    use_left = left_conf >= right_conf

    # Bilateral midpoints
    l_shoulder, r_shoulder = pt("left_shoulder"), pt("right_shoulder")
    l_hip, r_hip = pt("left_hip"), pt("right_hip")
    l_knee, r_knee = pt("left_knee"), pt("right_knee")
    l_elbow, r_elbow = pt("left_elbow"), pt("right_elbow")
    l_wrist, r_wrist = pt("left_wrist"), pt("right_wrist")
    l_ankle, r_ankle = pt("left_ankle"), pt("right_ankle")
    l_ear, r_ear = pt("left_ear"), pt("right_ear")
    nose = pt("nose")

    shoulder_mid = midpoint(l_shoulder, r_shoulder)
    hip_mid = midpoint(l_hip, r_hip)
    knee_mid = midpoint(l_knee, r_knee)
    elbow_mid = midpoint(l_elbow, r_elbow)
    # Head reference: prefer ear midpoint, fallback to nose
    ear_conf_sum = float(kps_conf[KP["left_ear"]]) + float(kps_conf[KP["right_ear"]])
    head = midpoint(l_ear, r_ear) if ear_conf_sum > 0.5 else nose

    return {
        # Midline joints (for normalization reference)
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "knee_mid": knee_mid,
        "elbow_mid": elbow_mid,
        "head": head,
        # Lateral side joints (for angle computation)
        "shoulder_side": l_shoulder if use_left else r_shoulder,
        "elbow_side": l_elbow if use_left else r_elbow,
        "wrist_side": l_wrist if use_left else r_wrist,
        "hip_side": l_hip if use_left else r_hip,
        "knee_side": l_knee if use_left else r_knee,
        "ankle_side": l_ankle if use_left else r_ankle,
    }


def normalize_keypoints(
    joints: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Convert pixel keypoints to body-relative coordinates:
      1. Translate: origin = hip_mid
      2. Scale:     unit = torso length (shoulder_mid → hip_mid distance)

    This makes features invariant to camera distance and person position in frame.
    Returns a new dict of normalized (x, y) values.
    """
    origin = joints["hip_mid"]
    torso_vec = _vec2(joints["hip_mid"], joints["shoulder_mid"])
    torso_len = _magnitude(torso_vec)

    if torso_len < 1e-3:
        # Degenerate case: return zeros
        return {k: (0.0, 0.0) for k in joints}

    norm: Dict[str, Tuple[float, float]] = {}
    for name, (x, y) in joints.items():
        nx = (x - origin[0]) / torso_len
        ny = (y - origin[1]) / torso_len
        norm[name] = (nx, ny)

    return norm


# ---------------------------------------------------------------------------
# Temporal Smoothing Buffer
# ---------------------------------------------------------------------------

class SmoothingBuffer:
    """
    Maintains a rolling window of scalar values (e.g., an angle) and
    provides moving-average smoothing.

    Args:
        window: number of frames to keep (default 8)
    """

    def __init__(self, window: int = 8) -> None:
        self._buf: deque = deque(maxlen=window)
        self._window = window

    def push(self, value: float) -> None:
        """Add a new value; drops oldest if at capacity."""
        if not math.isnan(value):
            self._buf.append(value)

    def mean(self) -> float:
        """Return moving average; NaN if buffer is empty."""
        if not self._buf:
            return float("nan")
        return float(np.mean(list(self._buf)))

    def velocity(self) -> float:
        """
        Approximate first derivative: current − previous value.
        Returns NaN if fewer than 2 samples.
        """
        if len(self._buf) < 2:
            return float("nan")
        return float(self._buf[-1] - self._buf[-2])

    def acceleration(self) -> float:
        """
        Approximate second derivative using last 3 values.
        Returns NaN if fewer than 3 samples.
        """
        if len(self._buf) < 3:
            return float("nan")
        v_curr = self._buf[-1] - self._buf[-2]
        v_prev = self._buf[-2] - self._buf[-3]
        return float(v_curr - v_prev)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Feature Extraction (Training & Inference Shared)
# ---------------------------------------------------------------------------

def extract_all_features(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    buffers: Dict[str, SmoothingBuffer],
) -> Optional[Dict[str, float]]:
    """
    Full feature extraction pipeline for one frame.

    Steps:
      1. Validate pose (essential joints confidence >= MIN_KP_CONF)
      2. Extract raw joints
      3. Compute ergonomic angles (vector-based)
      4. Push angles into smoothing buffers → moving-average smoothed angles
      5. Compute temporal features (velocity, acceleration)
      6. Normalize keypoints (hip-centered, torso-scaled)

    Args:
        kps_xy:   (17, 2) pixel coordinates from YOLOv8
        kps_conf: (17,)   per-joint confidence from YOLOv8
        buffers:  dict of angle_name → SmoothingBuffer (maintained externally)

    Returns:
        dict of feature_name → float value, or None if pose is invalid.

    Feature keys returned:
        back_angle, knee_angle, neck_angle, elbow_angle    (smoothed angles)
        back_vel, knee_vel, neck_vel                        (angle velocities)
        back_acc, knee_acc                                  (angle accelerations)
        norm_shoulder_x, norm_shoulder_y                    (normalized coords)
        norm_hip_x, norm_hip_y
        norm_knee_x, norm_knee_y
    """
    # Step 1: Validate
    if not is_pose_valid(kps_conf):
        return None

    # Step 2: Extract raw joints
    joints = extract_raw_joints(kps_xy, kps_conf)

    # Step 3: Compute raw ergonomic angles
    raw_back = calc_back_angle(joints["shoulder_mid"], joints["hip_mid"], joints["knee_mid"])
    raw_knee = knee_angle_flex(
        joints["hip_side"], joints["knee_side"], joints["ankle_side"]
    )
    raw_neck = neck_angle_tilt(
        joints["head"], joints["shoulder_mid"], joints["hip_mid"]
    )
    raw_elbow = elbow_angle_flex(
        joints["shoulder_side"], joints["elbow_side"], joints["wrist_side"]
    )

    # Reject if any primary angle is NaN
    if any(math.isnan(a) for a in [raw_back, raw_knee, raw_neck]):
        return None

    # Step 4: Push into buffers and read smoothed means
    for name, val in [
        ("back", raw_back),
        ("knee", raw_knee),
        ("neck", raw_neck),
        ("elbow", raw_elbow),
    ]:
        buffers[name].push(val)

    back_angle = buffers["back"].mean()
    knee_angle = buffers["knee"].mean()
    neck_angle = buffers["neck"].mean()
    elbow_angle = buffers["elbow"].mean()

    # Step 5: Temporal features
    back_vel = buffers["back"].velocity()
    knee_vel = buffers["knee"].velocity()
    neck_vel = buffers["neck"].velocity()
    back_acc = buffers["back"].acceleration()
    knee_acc = buffers["knee"].acceleration()

    # Replace NaN temporals with 0 (safe default for early frames)
    def safe(v: float) -> float:
        return 0.0 if math.isnan(v) else v

    # Step 6: Normalize keypoints
    norm = normalize_keypoints(joints)

    return {
        # --- Angles (smoothed, degrees) ---
        "back_angle":   back_angle,
        "knee_angle":   knee_angle,
        "neck_angle":   neck_angle,
        "elbow_angle":  elbow_angle if not math.isnan(elbow_angle) else 180.0,
        # --- Temporal features ---
        "back_vel":     safe(back_vel),
        "knee_vel":     safe(knee_vel),
        "neck_vel":     safe(neck_vel),
        "back_acc":     safe(back_acc),
        "knee_acc":     safe(knee_acc),
        # --- Normalized keypoint coordinates ---
        "norm_shoulder_x": norm["shoulder_mid"][0],
        "norm_shoulder_y": norm["shoulder_mid"][1],
        "norm_hip_x":      norm["hip_mid"][0],
        "norm_hip_y":      norm["hip_mid"][1],
        "norm_knee_x":     norm["knee_mid"][0],
        "norm_knee_y":     norm["knee_mid"][1],
    }


# ---------------------------------------------------------------------------
# Feature Vector Assembly
# ---------------------------------------------------------------------------

# Canonical ordered feature names used by the classifier (must match training).
# 9 stable features: 3 angles + 6 normalized coordinates.
# Velocity/acceleration deliberately excluded — they cause noise instability.
FEATURE_COLS: List[str] = [
    "back_angle", "knee_angle", "neck_angle",
    "norm_shoulder_x", "norm_shoulder_y",
    "norm_hip_x",      "norm_hip_y",
    "norm_knee_x",     "norm_knee_y",
]


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Assemble the canonical feature vector for classifier inference.

    Args:
        features: dict returned by extract_all_features()

    Returns:
        np.ndarray of shape (1, 9) as float32
    """
    vec = [features.get(col, 0.0) for col in FEATURE_COLS]
    return np.array([vec], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hybrid Rule + ML Classification  (Part 10 of specification)
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


def make_empty_buffers() -> Dict[str, SmoothingBuffer]:
    """Create a fresh set of SmoothingBuffers for all angle channels."""
    return {
        "back":  SmoothingBuffer(window=8),
        "knee":  SmoothingBuffer(window=8),
        "neck":  SmoothingBuffer(window=8),
        "elbow": SmoothingBuffer(window=8),
    }
