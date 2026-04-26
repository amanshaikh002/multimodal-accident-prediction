"""
combined_service.py
===================
Combined PPE + Pose safety analysis service.

Architecture
------------
Three phases per request:
  1. PPE-only pass  → ppe_metrics   + combined_ppe_annotated.mp4
  2. Pose-only pass → pose_metrics  + combined_pose_annotated.mp4
  3. Merge render   → combined_annotated.mp4  (skeleton + PPE boxes + banner)

Phase 3 is the key addition: it processes the video frame-by-frame, running
BOTH models on the SAME frame in order:
    a) pose model  → results[0].plot()  (draws skeleton automatically)
    b) PPE model   → raw detections drawn on top of the skeleton frame
    c) Final status banner drawn at the top of the merged frame

This guarantees:
  - Skeleton is always visible (YOLO handles drawing internally)
  - PPE boxes appear on top of the skeleton without overwriting it
  - A single merged output video is returned to the frontend

Output JSON schema
------------------
{
    "mode":            "combined",
    "final_status":    "HIGH RISK" | "UNSAFE" | "MODERATE" | "SAFE",
    "ppe_status":      "SAFE" | "UNSAFE",
    "pose_status":     "SAFE" | "MODERATE" | "UNSAFE",
    "ppe_score":       float,
    "pose_score":      float,
    "total_frames":    int,
    "violations":      [{frame, type, reason, severity}],
    "summary_message": str,
    "recommendations": [str],
    "video_output":    str        # served by /output/
}
"""

import logging
import os
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from services.ppe_service  import run_ppe_detection
from services.pose_service import process_pose_video

from utils.ppe_utils import (
    bbox_to_list, evaluate_frame_safety, is_ppe_on_person, normalize_label,
)
from utils.pose_utils import (
    FEATURE_COLS, LABEL_NAMES, LABEL_COLORS_BGR,
    build_feature_vector, extract_all_features, hybrid_classify,
    is_pose_valid, make_empty_buffers, select_primary_person,
)
from utils.video_utils import finalize_video, get_video_writer, open_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

_HERE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

_YOLO_POSE_PATH: str = os.path.join(_HERE, "models", "yolov8s-pose.pt")
_YOLO_PPE_PATH:  str = os.path.join(_HERE, "models", "ppe_model.pt")
_POSE_CLF_PATH:  str = os.path.join(_HERE, "models", "model.pkl")

_CONF_POSE:     float = 0.25
_CONF_PPE:      float = 0.45
_FRAME_STRIDE:  int   = 2
_INFER_W:       int   = 640
_INFER_H:       int   = 480
_SMOOTH_WINDOW: int   = 5

_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model singletons (shared across requests)
# ---------------------------------------------------------------------------

_pose_yolo: Optional[YOLO] = None
_ppe_yolo:  Optional[YOLO] = None
_pose_clf:  Optional[Any]  = None


def _get_pose_yolo() -> YOLO:
    global _pose_yolo
    if _pose_yolo is None:
        src = _YOLO_POSE_PATH if os.path.isfile(_YOLO_POSE_PATH) else "yolov8s-pose.pt"
        logger.info("[COMBINED] Loading pose YOLO: %s", src)
        _pose_yolo = YOLO(src)
    return _pose_yolo


def _get_ppe_yolo() -> YOLO:
    global _ppe_yolo
    if _ppe_yolo is None:
        if not os.path.isfile(_YOLO_PPE_PATH):
            raise FileNotFoundError(
                f"PPE model not found: '{_YOLO_PPE_PATH}'. "
                "Copy ppe_model.pt into backend/models/."
            )
        logger.info("[COMBINED] Loading PPE YOLO: %s", _YOLO_PPE_PATH)
        _ppe_yolo = YOLO(_YOLO_PPE_PATH)
        logger.info("[COMBINED] PPE classes: %s", list(_ppe_yolo.names.values()))
    return _ppe_yolo


def _get_pose_clf():
    global _pose_clf
    if _pose_clf is None:
        import joblib
        if not os.path.isfile(_POSE_CLF_PATH):
            raise FileNotFoundError(
                f"Pose classifier not found: '{_POSE_CLF_PATH}'. "
                "Run train_model.py first."
            )
        _pose_clf = joblib.load(_POSE_CLF_PATH)
        logger.info("[COMBINED] Pose classifier loaded.")
    return _pose_clf


# NOTE: Models are NOT pre-loaded at import time.
# All three models (pose YOLO, PPE YOLO, pose classifier) are loaded lazily
# on the first request via their singleton getters. This keeps uvicorn startup
# fast and avoids the ~3-minute GPU-initialization timeout.


# ---------------------------------------------------------------------------
# Clean visualization constants (BGR)
# ---------------------------------------------------------------------------

_C_SAFE     = (0,   210,   0)      # green
_C_MODERATE = (0,   200, 255)      # amber-yellow
_C_UNSAFE   = (0,    60, 220)      # red
_C_HIGHRISK = (0,    0,  180)      # deep red
_C_SKELETON = (230, 230,   0)      # cyan-gold skeleton lines
_C_JOINT    = (255, 255, 255)      # white joint dots
_C_WHITE    = (255, 255, 255)
_C_BLACK    = (0,     0,   0)

# COCO-17 skeleton bone pairs (matches pose_utils.SKELETON)
_SKELETON_PAIRS = [
    (5, 6),  (5, 7),   (7, 9),
    (6, 8),  (8, 10),  (5, 11),
    (6, 12), (11, 12), (11, 13),
    (13, 15),(12, 14), (14, 16),
    (0, 5),  (0, 6),
]

_STATUS_COLOUR = {
    "SAFE":      _C_SAFE,
    "MODERATE":  _C_MODERATE,
    "UNSAFE":    _C_UNSAFE,
    "HIGH RISK": _C_HIGHRISK,
    "UNKNOWN":   (120, 120, 120),
}


def _draw_skeleton_clean(
    frame: np.ndarray,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_thr: float = 0.30,
) -> None:
    """Draw a clean single-color skeleton on frame IN-PLACE."""
    # Draw bone lines
    for a, b in _SKELETON_PAIRS:
        if kps_conf[a] >= conf_thr and kps_conf[b] >= conf_thr:
            p1 = (int(kps_xy[a][0]), int(kps_xy[a][1]))
            p2 = (int(kps_xy[b][0]), int(kps_xy[b][1]))
            cv2.line(frame, p1, p2, _C_SKELETON, 2, cv2.LINE_AA)
    # Draw joint dots
    for i in range(len(kps_xy)):
        if kps_conf[i] >= conf_thr:
            cx, cy = int(kps_xy[i][0]), int(kps_xy[i][1])
            cv2.circle(frame, (cx, cy), 3, _C_JOINT, -1, cv2.LINE_AA)


def _draw_person_box(
    frame: np.ndarray,
    box_xyxy: List[float],
    pose_label: str,
    helmet_ok: bool,
    vest_ok: bool,
    final_status: str,
) -> None:
    """
    Draw ONE clean bounding box per person with:
      - Color-coded border (green/amber/red)
      - Status label above the box
      - PPE icons inside the top of the box (no separate PPE boxes)
    """
    colour = _STATUS_COLOUR.get(final_status, _C_UNSAFE)
    x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── 1. Bounding box border (2 px, color-coded) ───────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

    # ── 2. Status badge above the box ────────────────────────────────────────
    status_text = final_status
    (sw, sh), _ = cv2.getTextSize(status_text, font, 0.60, 2)
    badge_y1 = max(0, y1 - sh - 10)
    badge_y2 = y1
    cv2.rectangle(frame, (x1, badge_y1), (x1 + sw + 12, badge_y2), colour, -1)
    cv2.putText(frame, status_text, (x1 + 6, badge_y2 - 4),
                font, 0.60, _C_WHITE, 2, cv2.LINE_AA)

    # ── 3. PPE status line inside top of box ─────────────────────────────────
    helmet_str = "Helmet: OK " if helmet_ok else "Helmet: NO "
    vest_str   = "| Vest: OK"  if vest_ok   else "| Vest: NO"
    ppe_text   = helmet_str + vest_str

    # Semi-transparent background strip for readability
    strip_h = 20
    strip_y1, strip_y2 = y1 + 2, y1 + 2 + strip_h
    if strip_y2 < y2:  # only draw if box is tall enough
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, strip_y1), (x2, strip_y2), _C_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        helmet_col = _C_SAFE if helmet_ok else _C_UNSAFE
        vest_col   = _C_SAFE if vest_ok   else _C_UNSAFE
        # Draw helmet text in its own color
        cv2.putText(frame, helmet_str, (x1 + 4, strip_y2 - 4),
                    font, 0.44, helmet_col, 1, cv2.LINE_AA)
        (htw, _), _ = cv2.getTextSize(helmet_str, font, 0.44, 1)
        cv2.putText(frame, vest_str, (x1 + 4 + htw, strip_y2 - 4),
                    font, 0.44, vest_col, 1, cv2.LINE_AA)

    # ── 4. Posture label (small, bottom of box) ───────────────────────────────
    pose_colour = _STATUS_COLOUR.get(pose_label, (150, 150, 150))
    pose_text   = f"Posture: {pose_label}"
    (ptw, pth), _ = cv2.getTextSize(pose_text, font, 0.42, 1)
    pt_x, pt_y = x1 + 4, min(y2 - 4, y2 - 4)
    if pt_y - pth > y1 + strip_h + 4:
        cv2.putText(frame, pose_text, (pt_x, pt_y),
                    font, 0.42, pose_colour, 1, cv2.LINE_AA)


def _draw_global_banner(
    frame: np.ndarray,
    worst_status: str,
    n_persons: int,
    all_missing: List[str],
    worst_pose: str,
) -> None:
    """
    Draw a single clean global status banner at the very top of the frame.
    Layout:  [STATUS]  |  <reason text>  |  N person(s)
    """
    colour = _STATUS_COLOUR.get(worst_status, _C_UNSAFE)

    # Build reason string
    parts = []
    if all_missing:
        parts.append("No " + " + No ".join(m.capitalize() for m in sorted(set(all_missing))))
    if worst_pose in ("MODERATE", "UNSAFE", "HIGH RISK"):
        parts.append(f"Posture {worst_pose}")
    if not parts:
        parts = ["All compliant"]

    reason = "  ·  ".join(parts)
    text   = f"{worst_status}  |  {reason}  |  {n_persons} person(s)"

    # Solid banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), colour, -1)
    # Subtle bottom edge
    cv2.line(frame, (0, 38), (frame.shape[1], 38), _C_WHITE, 1)
    cv2.putText(frame, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _C_WHITE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main rendering pass
# ---------------------------------------------------------------------------

def _render_combined_video(
    video_path: str,
    output_path: str,
) -> None:
    """
    Single-pass rendering producing a CLEAN professional combined video:

    For every frame:
      1. Run pose YOLO    → raw keypoints + person bboxes (NO .plot())
      2. Run PPE  YOLO    → raw detections on original frame
      3. Per person:
           a) classify posture via hybrid_classify
           b) find which PPE items overlap THIS person's bbox
           c) draw clean skeleton (cyan lines, white dots)
           d) draw ONE color-coded person box
           e) show PPE status text inside box (no separate PPE boxes)
      4. Draw global banner at top (worst combined status across all people)
    """
    pose_yolo = _get_pose_yolo()
    ppe_yolo  = _get_ppe_yolo()
    clf       = _get_pose_clf()

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer: Optional[cv2.VideoWriter] = None

    # Per-person pose smoothing buffers (keyed by person index)
    person_buffers: Dict[int, Any] = {}
    person_preds:   Dict[int, deque] = {}

    raw_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_idx += 1
            if raw_idx % _FRAME_STRIDE != 0:
                continue

            frame = cv2.resize(frame, (_INFER_W, _INFER_H), interpolation=cv2.INTER_LINEAR)
            canvas = frame.copy()   # all drawing goes on canvas; frame stays clean for PPE inference

            # ── STEP 1: Pose YOLO ────────────────────────────────────────────
            try:
                pose_res = pose_yolo.predict(
                    frame, verbose=False, conf=_CONF_POSE,
                    imgsz=_INFER_W, device=_DEVICE,
                )
                r = pose_res[0]
                have_pose = (
                    r.keypoints is not None and
                    r.boxes     is not None and
                    len(r.boxes) > 0
                )
            except Exception as e:
                logger.warning("[COMBINED] Pose YOLO error frame %d: %s", raw_idx, e)
                have_pose = False
                r = None

            # ── STEP 2: PPE YOLO (on original clean frame) ───────────────────
            ppe_dets: List[Dict] = []
            try:
                ppe_res = ppe_yolo(frame, device=_DEVICE, verbose=False)
                for res in ppe_res:
                    if res.boxes is None:
                        continue
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        if conf < _CONF_PPE:
                            continue
                        lbl  = normalize_label(res.names[int(box.cls[0])])
                        bbox = bbox_to_list(box.xyxy[0])
                        ppe_dets.append({
                            "label": lbl, "confidence": round(conf, 4), "bbox": bbox,
                        })
                logger.debug(
                    "[COMBINED] frame %d — pose detected: %s  ppe dets: %d",
                    raw_idx, have_pose, len(ppe_dets),
                )
                # Debug: log which PPE items found
                found_ppe = [d["label"] for d in ppe_dets]
                logger.debug("[COMBINED] PPE labels: %s", found_ppe)
            except Exception as e:
                logger.warning("[COMBINED] PPE YOLO error frame %d: %s", raw_idx, e)

            # Separate PPE items for overlap checking
            ppe_helmet_boxes = [d["bbox"] for d in ppe_dets if d["label"] == "helmet"]
            ppe_vest_boxes   = [d["bbox"] for d in ppe_dets if d["label"] == "vest"]

            # ── STEP 3: Per-person processing ────────────────────────────────
            worst_status  = "SAFE"
            worst_pose    = "SAFE"
            all_missing: List[str] = []
            n_persons = 0

            if have_pose:
                boxes_np     = r.boxes.xyxy.cpu().numpy()    # (N,4)
                kps_xy_all   = r.keypoints.xy.cpu().numpy()  # (N,17,2)
                kps_conf_all = r.keypoints.conf.cpu().numpy()# (N,17)
                n_persons    = len(boxes_np)

                for pid in range(n_persons):
                    person_box = boxes_np[pid].tolist()      # [x1,y1,x2,y2]
                    kps_xy     = kps_xy_all[pid]
                    kps_conf   = kps_conf_all[pid]

                    # ── Posture classification ────────────────────────────────
                    pose_label = "UNKNOWN"
                    confidence = 0.0
                    if is_pose_valid(kps_conf):
                        if pid not in person_buffers:
                            person_buffers[pid] = make_empty_buffers()
                            person_preds[pid]   = deque(maxlen=_SMOOTH_WINDOW)

                        feats = extract_all_features(kps_xy, kps_conf, person_buffers[pid])
                        if feats is not None:
                            x_vec = build_feature_vector(feats)
                            if not np.isnan(x_vec).any():
                                raw_pred, confidence, _ = hybrid_classify(feats, clf, x_vec)
                                person_preds[pid].append(raw_pred)
                                smoothed   = Counter(person_preds[pid]).most_common(1)[0][0]
                                pose_label = LABEL_NAMES.get(smoothed, "UNKNOWN")

                    # ── PPE overlap check for THIS person ─────────────────────
                    helmet_ok = any(
                        is_ppe_on_person(person_box, hb)
                        for hb in ppe_helmet_boxes
                    )
                    vest_ok = any(
                        is_ppe_on_person(person_box, vb)
                        for vb in ppe_vest_boxes
                    )

                    logger.debug(
                        "[COMBINED] person %d: pose=%s  helmet=%s  vest=%s",
                        pid, pose_label, helmet_ok, vest_ok,
                    )

                    person_missing = []
                    if not helmet_ok: person_missing.append("helmet")
                    if not vest_ok:   person_missing.append("vest")
                    all_missing.extend(person_missing)

                    # ── Per-person combined status ────────────────────────────
                    if person_missing and pose_label == "UNSAFE":
                        person_status = "HIGH RISK"
                    elif person_missing or pose_label == "UNSAFE":
                        person_status = "UNSAFE"
                    elif pose_label == "MODERATE":
                        person_status = "MODERATE"
                    else:
                        person_status = "SAFE"

                    # Track worst status across all persons
                    _rank = {"SAFE": 0, "MODERATE": 1, "UNSAFE": 2, "HIGH RISK": 3, "UNKNOWN": 0}
                    if _rank.get(person_status, 0) > _rank.get(worst_status, 0):
                        worst_status = person_status
                    if _rank.get(pose_label, 0) > _rank.get(worst_pose, 0):
                        worst_pose = pose_label

                    # ── Draw clean skeleton ───────────────────────────────────
                    _draw_skeleton_clean(canvas, kps_xy, kps_conf)

                    # ── Draw ONE color-coded person box + PPE status ──────────
                    _draw_person_box(
                        canvas, person_box,
                        pose_label, helmet_ok, vest_ok, person_status,
                    )

            elif not have_pose:
                # No person detected — just evaluate PPE globally
                _, global_missing = evaluate_frame_safety(ppe_dets)
                all_missing = global_missing
                worst_status = "UNSAFE" if global_missing else "SAFE"
                n_persons    = 0

            # ── STEP 4: Global banner (drawn last so it's always on top) ─────
            _draw_global_banner(canvas, worst_status, n_persons, all_missing, worst_pose)

            # ── STEP 5: Init writer on first frame ────────────────────────────
            if writer is None:
                try:
                    writer = get_video_writer(
                        output_path,
                        fps    = max(1.0, fps / _FRAME_STRIDE),
                        width  = _INFER_W,
                        height = _INFER_H,
                    )
                except RuntimeError as e:
                    logger.error("[COMBINED] Cannot create video writer: %s", e)
                    break

            writer.write(canvas)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            finalize_video(output_path)

    logger.info("[COMBINED] Clean merged video → %s", output_path)


# ---------------------------------------------------------------------------
# Recommendation & message helpers
# ---------------------------------------------------------------------------

_PPE_RECS: Dict[str, str] = {
    "helmet":  "Wear a certified hard-hat at all times on site.",
    "vest":    "Wear a high-visibility safety vest to remain visible to machinery.",
    "gloves":  "Use rated protective gloves to prevent hand injuries.",
    "boots":   "Steel-toe boots must be worn to protect against falling objects.",
}

_POSE_RECS: Dict[str, str] = {
    "back":    "Keep the back straight; bend at the knees and hips, not the waist.",
    "knee":    "Bend knees more when lifting to reduce spinal compression.",
    "neck":    "Align head with the spine; raise the work surface to reduce neck strain.",
    "default": "Maintain proper lifting posture: straight back, bent knees, load close to body.",
}


def _ppe_msg(missing: List[str]) -> str:
    if not missing:
        return ""
    labels = [m.capitalize() for m in missing]
    return f"Worker is not wearing {', '.join(labels[:-1])} and {labels[-1]}" if len(labels) > 1 \
        else f"Worker is not wearing a {labels[0]}"


def _pose_msg(reason: str) -> str:
    r = reason.lower()
    if "back" in r:
        return "Worker is bending their back excessively while lifting"
    if "knee" in r or "stiff" in r or "leg" in r:
        return "Worker is lifting with locked or stiff knees"
    if "neck" in r:
        return "Worker is tilting their head forward excessively"
    return reason


def _ppe_recs(all_missing: List[str]) -> List[str]:
    seen, recs = set(), []
    for item in all_missing:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            recs.append(_PPE_RECS.get(key, f"Ensure {item} is worn per site safety protocol."))
    return recs


def _pose_recs(pose_violations: List[Dict]) -> List[str]:
    seen, recs = set(), []
    for v in pose_violations:
        r = (v.get("reason") or v.get("issue") or "").lower()
        if "back" in r and "back" not in seen:
            seen.add("back"); recs.append(_POSE_RECS["back"])
        elif ("knee" in r or "stiff" in r) and "knee" not in seen:
            seen.add("knee"); recs.append(_POSE_RECS["knee"])
        elif "neck" in r and "neck" not in seen:
            seen.add("neck"); recs.append(_POSE_RECS["neck"])
    if not recs and pose_violations:
        recs.append(_POSE_RECS["default"])
    return recs


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------

def _ppe_status(r: Dict) -> str:
    return "SAFE" if r.get("compliance_score", 0.0) >= 70.0 else "UNSAFE"


def _pose_status(r: Dict) -> str:
    s = r.get("safety_score", 0.0)
    if s >= 80.0: return "SAFE"
    if s >= 50.0: return "MODERATE"
    return "UNSAFE"


def _final_status(ppe_st: str, pose_st: str) -> str:
    if ppe_st == "UNSAFE" and pose_st == "UNSAFE": return "HIGH RISK"
    if ppe_st == "UNSAFE" or pose_st == "UNSAFE":  return "UNSAFE"
    if pose_st == "MODERATE":                       return "MODERATE"
    return "SAFE"


def get_final_status(ppe_status: str, pose_status: str, fire_status: str) -> str:
    """
    Unified Decision Engine — derives the single worst-case status
    across all three modules.

    Priority (highest → lowest):
        CRITICAL  — fire is UNSAFE (immediate evacuation)
        HIGH RISK — both PPE and Pose are UNSAFE
        UNSAFE    — either PPE or Pose is UNSAFE
        MODERATE  — PPE is SAFE but Pose is MODERATE
        SAFE      — all modules report SAFE
    """
    if fire_status == "UNSAFE":
        return "CRITICAL"
    if ppe_status == "UNSAFE" and pose_status == "UNSAFE":
        return "HIGH RISK"
    if ppe_status == "UNSAFE" or pose_status == "UNSAFE":
        return "UNSAFE"
    if ppe_status == "MODERATE" or pose_status == "MODERATE":
        return "MODERATE"
    return "SAFE"


# ---------------------------------------------------------------------------
# Merge JSON results
# ---------------------------------------------------------------------------

def merge_results(ppe_result: Dict, pose_result: Dict) -> Dict:
    ppe_st  = _ppe_status(ppe_result)
    pose_st = _pose_status(pose_result)
    final   = _final_status(ppe_st, pose_st)

    ppe_v  = ppe_result.get("violations",  [])
    pose_v = pose_result.get("violations", [])

    violations: List[Dict] = []
    for v in ppe_v:
        for item in (v.get("missing") or []):
            violations.append({
                "frame":    v.get("frame", "?"),
                "type":     "PPE",
                "reason":   f"No {item.capitalize()}",
                "severity": "high",
            })
    for v in pose_v:
        reason = v.get("reason") or v.get("issue") or "Bad posture"
        violations.append({
            "frame":    v.get("frame", "?"),
            "type":     "POSE",
            "reason":   _pose_msg(reason),
            "severity": "high" if pose_st == "UNSAFE" else "medium",
        })

    all_missing = list({item for v in ppe_v for item in (v.get("missing") or [])})
    recommendations = _ppe_recs(all_missing) + _pose_recs(pose_v)
    if not recommendations:
        recommendations = ["No critical recommendations — continue monitoring."]

    ppe_issues  = list({item for v in ppe_v for item in (v.get("missing") or [])})
    pose_issues = list({(v.get("reason") or v.get("issue") or "bad posture") for v in pose_v})
    parts = (
        ([_ppe_msg(ppe_issues)]  if ppe_issues  else []) +
        ([_pose_msg(pose_issues[0])] if pose_issues else [])
    )
    message = (
        " and ".join(parts) + "." if parts
        else "No safety violations detected — workspace appears compliant."
    )

    return {
        "mode":            "combined",
        "final_status":    final,
        "ppe_status":      ppe_st,
        "pose_status":     pose_st,
        "ppe_score":       ppe_result.get("compliance_score", 0.0),
        "pose_score":      pose_result.get("safety_score",    0.0),
        "total_frames":    max(
            ppe_result.get("total_frames",  0),
            pose_result.get("total_frames", 0),
        ),
        "violations":      violations,
        "summary_message": message,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_combined_video(
    video_path:        str,
    ppe_output_path:   str,   # kept for backward compat (not written)
    pose_output_path:  str,   # the merged combined video is written HERE
) -> Dict[str, Any]:
    """
    Full combined pipeline:
      1. PPE-only pass  → metrics JSON
      2. Pose-only pass → metrics JSON
      3. Merged render  → single annotated video (skeleton + PPE + banner)
    """
    # ── Phase 1: PPE metrics ─────────────────────────────────────────────────
    logger.info("[COMBINED] Phase 1 — PPE metrics pass...")
    ppe_result = run_ppe_detection(
        video_path        = video_path,
        output_video_path = None,   # no separate PPE video needed
    )
    logger.info(
        "[COMBINED] PPE done — compliance %.1f%%  violations: %d",
        ppe_result.get("compliance_score", 0),
        len(ppe_result.get("violations", [])),
    )

    # ── Phase 2: Pose metrics ─────────────────────────────────────────────────
    logger.info("[COMBINED] Phase 2 — Pose metrics pass...")
    pose_result = process_pose_video(
        video_path        = video_path,
        output_video_path = None,   # no separate pose video needed
    )
    logger.info(
        "[COMBINED] Pose done — safety %.1f%%  violations: %d",
        pose_result.get("safety_score", 0),
        len(pose_result.get("violations", [])),
    )

    # ── Phase 3: Merged visualization ────────────────────────────────────────
    logger.info("[COMBINED] Phase 3 — merged render (skeleton + PPE boxes)...")
    _render_combined_video(video_path, pose_output_path)

    # ── Build and return merged result ────────────────────────────────────────
    combined = merge_results(ppe_result, pose_result)
    logger.info(
        "[COMBINED] Final status: %s  (PPE=%s  Pose=%s)",
        combined["final_status"], combined["ppe_status"], combined["pose_status"],
    )
    return combined
