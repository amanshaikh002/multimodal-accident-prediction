"""
app_yolo.py — Worker Pose Safety Monitoring System
===================================================
Production-Level Streamlit Inference Application

Full pipeline (YOLOv8-only, no MediaPipe):
  1. Upload MP4 video
  2. YOLOv8s-pose detects keypoints frame-by-frame
  3. Select primary person (largest bbox + closest to center)
  4. Extract 15 features via utils.extract_all_features()
  5. Classify posture with XGBoost model
  6. Temporal sliding window (8 frames) majority vote
  7. Draw rich overlay: skeleton, bounding box, angles, status badge
  8. Smart voice alert: UNSAFE + confidence>75% + ≥10 consecutive frames
  9. Display real-time angle graph (Plotly)

Usage:
  streamlit run app_yolo.py
"""

import math
import subprocess
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import streamlit as st
from ultralytics import YOLO

from utils import (
    KP,
    SKELETON,
    LABEL_NAMES,
    LABEL_COLORS_BGR,
    FEATURE_COLS,
    make_empty_buffers,
    select_primary_person,
    extract_all_features,
    build_feature_vector,
    hybrid_classify,
    is_pose_valid,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_YOLO_MODEL: str = "yolov8s-pose.pt"
FRAME_W: int = 640
FRAME_H: int = 480
SMOOTHING_WINDOW: int = 8          # temporal majority vote window
UNSAFE_TRIGGER_FRAMES: int = 10    # consecutive UNSAFE frames for voice alert
CONFIDENCE_THRESHOLD: float = 0.75 # min confidence to trigger voice alert
ALERT_COOLDOWN: float = 5.0        # seconds between voice alerts
YOLO_CONF: float = 0.25            # YOLO detection confidence threshold

# CSS color map for Streamlit UI
STATUS_COLORS = {
    "SAFE": "#00c853",
    "MODERATE": "#ffd600",
    "UNSAFE": "#d50000",
}

# Track active TTS subprocess handles (module-level to survive reruns)
_TTS_PROCS: List[subprocess.Popen] = []


# ---------------------------------------------------------------------------
# Voice Alert System (pyttsx3 subprocess — avoids COM / threading crashes)
# ---------------------------------------------------------------------------

def _contextual_alert_text(features: Dict[str, float]) -> str:
    """Build a targeted voice message based on which angles are worst.
    
    Uses VERTEX ANGLE convention: 180° = straight/upright, lower = more bent.
    """
    parts: List[str] = ["Unsafe posture detected."]
    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    if not math.isnan(ba) and ba < 130:
        parts.append("Avoid bending your back excessively.")
    if not math.isnan(ka) and ka > 155:
        parts.append("Bend your knees more when lifting.")
    if not math.isnan(na) and na < 130:
        parts.append("Keep your head aligned with your spine.")

    return " ".join(parts)


def trigger_voice_alert(text: str) -> None:
    """
    Spawn pyttsx3 TTS in an independent subprocess.
    Non-blocking — does not stall the Streamlit video loop.
    """
    global _TTS_PROCS
    code = (
        "import pyttsx3\n"
        "try:\n"
        "    e = pyttsx3.init()\n"
        "    e.setProperty('rate', 165)\n"
        "    e.setProperty('volume', 1.0)\n"
        f"    e.say({repr(text)})\n"
        "    e.runAndWait()\n"
        "except Exception:\n"
        "    pass\n"
    )
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        p = subprocess.Popen([sys.executable, "-c", code], creationflags=flags)
        _TTS_PROCS.append(p)
        # Prune completed processes
        _TTS_PROCS = [proc for proc in _TTS_PROCS if proc.poll() is None]
    except Exception as exc:
        print(f"[TTS] Subprocess failed: {exc}")


def kill_all_tts() -> None:
    """Terminate any running TTS processes (cleanup on session end)."""
    global _TTS_PROCS
    for p in _TTS_PROCS:
        try:
            p.terminate()
        except Exception:
            pass
    _TTS_PROCS.clear()


# ---------------------------------------------------------------------------
# Drawing Helpers
# ---------------------------------------------------------------------------

def draw_skeleton(
    frame: np.ndarray,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_thr: float = 0.30,
) -> None:
    """Draw COCO-17 skeleton lines and joint circles on the frame."""
    for a, b in SKELETON:
        if float(kps_conf[a]) >= conf_thr and float(kps_conf[b]) >= conf_thr:
            p1 = (int(kps_xy[a][0]), int(kps_xy[a][1]))
            p2 = (int(kps_xy[b][0]), int(kps_xy[b][1]))
            cv2.line(frame, p1, p2, (255, 230, 50), 2, cv2.LINE_AA)
    for i in range(len(kps_xy)):
        if float(kps_conf[i]) >= conf_thr:
            cv2.circle(frame, (int(kps_xy[i][0]), int(kps_xy[i][1])), 4, (0, 255, 255), -1, cv2.LINE_AA)


def draw_status_overlay(
    frame: np.ndarray,
    features: Dict[str, float],
    label: str,
    confidence: float,
    reasons: List[str],
    box_xyxy: np.ndarray,
    color_bgr: Tuple[int, int, int],
) -> None:
    """
    Draw a rich HUD overlay on the frame:
      - Colored bounding box with label + confidence badge
      - Angle readouts (back / knee / neck) in top-left corner
      - Reason text below bounding box
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])

    # --- Bounding box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

    # --- Label badge (above the bounding box) ---
    badge_text = f"{label}  {confidence*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.65, 2)
    badge_y = max(y1 - 10, th + 8)
    cv2.rectangle(frame, (x1, badge_y - th - 6), (x1 + tw + 10, badge_y + 4), color_bgr, -1)
    cv2.putText(frame, badge_text, (x1 + 5, badge_y - 2), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # --- Angle HUD (top-left corner, semi-transparent background) ---
    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    hud_lines = [
        f"Back  : {ba:5.1f}deg" if not math.isnan(ba) else "Back  : N/A",
        f"Knee  : {ka:5.1f}deg" if not math.isnan(ka) else "Knee  : N/A",
        f"Neck  : {na:5.1f}deg" if not math.isnan(na) else "Neck  : N/A",
    ]
    hud_x, hud_y0 = 12, 28
    hud_h = 20
    hud_bg_w = 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + hud_bg_w, hud_y0 + len(hud_lines) * hud_h + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (hud_x, hud_y0 + i * hud_h), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # --- Reason text below the bounding box ---
    if reasons:
        reason_text = " | ".join(reasons[:2])  # max 2 reasons to avoid overflow
        (rw, rh), _ = cv2.getTextSize(reason_text, font, 0.48, 1)
        ry = min(y2 + rh + 8, FRAME_H - 6)
        cv2.rectangle(frame, (x1, ry - rh - 4), (x1 + rw + 8, ry + 4), (0, 0, 0), -1)
        cv2.putText(frame, reason_text, (x1 + 4, ry), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def build_reasons(features: Dict[str, float], label: str) -> List[str]:
    """Generate human-readable posture feedback strings.
    
    Uses VERTEX ANGLE convention: 180° = straight/upright, lower = more bent.
    """
    reasons: List[str] = []
    ba = features.get("back_angle", float("nan"))
    ka = features.get("knee_angle", float("nan"))
    na = features.get("neck_angle", float("nan"))

    if not math.isnan(ba):
        if ba < 120:
            reasons.append("Excessive back bending")
        elif ba < 145:
            reasons.append("Moderate back lean")

    if not math.isnan(ka):
        if ka > 165 and label in ("UNSAFE", "MODERATE"):
            reasons.append("Stiff legs — bend knees when lifting")
        elif ka < 100:
            reasons.append("Deep knee bend — use support")

    if not math.isnan(na):
        if na < 130:
            reasons.append("Significant neck tilt")
        elif na < 150:
            reasons.append("Slight neck forward")

    # Positive feedback
    if not math.isnan(ba) and not math.isnan(ka):
        if ba > 165 and ka > 155:
            reasons.append("Good upright posture")
        elif ba > 155 and ka < 130:
            reasons.append("Good ergonomic lifting posture")

    if not reasons:
        reasons.append({
            "SAFE": "Posture within safe limits",
            "MODERATE": "Posture risk — adjust position",
            "UNSAFE": "Unsafe posture — correct immediately",
        }.get(label, ""))

    return reasons


# ---------------------------------------------------------------------------
# Model & YOLO Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_classifier(model_path: str):
    return joblib.load(model_path)


@st.cache_resource
def load_yolo(model_name: str):
    return YOLO(model_name)


# ---------------------------------------------------------------------------
# Core Video Processing
# ---------------------------------------------------------------------------

def process_video(
    input_path: Path,
    output_path: Path,
    classifier,
    yolo_model,
    conf_threshold: float,
    smoothing_window: int,
) -> Dict:
    """
    Frame-by-frame inference loop.

    Returns a summary dict with angle history for the graph.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (FRAME_W, FRAME_H),
    )

    # --- Streamlit UI elements ---
    progress = st.progress(0.0)
    status_txt = st.empty()
    frame_placeholder = st.empty()

    st.markdown("---")
    col_status, col_conf = st.columns([2, 1])
    status_badge = col_status.empty()
    conf_meter = col_conf.empty()
    reason_box = st.empty()
    alert_box = st.empty()

    st.markdown("**Real-time Angle History**")
    angle_chart = st.empty()

    # --- State ---
    buffers = make_empty_buffers()
    recent_preds: deque = deque(maxlen=smoothing_window)
    consecutive_unsafe: int = 0
    last_alert_time: float = 0.0

    angle_history: Dict[str, List[float]] = {
        "back_angle": [], "knee_angle": [], "neck_angle": [], "frame": []
    }

    last_valid_features: Optional[Dict[str, float]] = None
    frame_idx: int = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize to standard resolution
            frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

            # --- YOLO inference ---
            results = yolo_model.predict(frame, verbose=False, conf=YOLO_CONF, imgsz=FRAME_W)
            result = results[0]

            no_pose = (
                result.keypoints is None
                or result.boxes is None
                or len(result.boxes) == 0
            )

            if no_pose:
                _write_no_pose(frame, writer, frame_placeholder, "No person detected")
                _update_progress(progress, status_txt, frame_idx, total_frames, "No person detected")
                frame_idx += 1
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            kps_xy_all = result.keypoints.xy.cpu().numpy()
            kps_conf_all = result.keypoints.conf.cpu().numpy()

            person_idx = select_primary_person(boxes, FRAME_W, FRAME_H)
            if person_idx is None:
                _write_no_pose(frame, writer, frame_placeholder, "No primary person")
                _update_progress(progress, status_txt, frame_idx, total_frames, "No primary person")
                frame_idx += 1
                continue

            kps_xy = kps_xy_all[person_idx]
            kps_conf = kps_conf_all[person_idx]
            person_box = boxes[person_idx]

            # Draw skeleton regardless of feature extraction result
            draw_skeleton(frame, kps_xy, kps_conf)

            # --- Feature extraction ---
            features = extract_all_features(kps_xy, kps_conf, buffers)

            if features is None:
                _write_no_pose(frame, writer, frame_placeholder, "Low keypoint confidence")
                _update_progress(progress, status_txt, frame_idx, total_frames, "Low confidence")
                frame_idx += 1
                continue

            last_valid_features = features

            # --- Build feature vector ---
            x_vec = build_feature_vector(features)

            if np.isnan(x_vec).any():
                _write_no_pose(frame, writer, frame_placeholder, "NaN in features")
                frame_idx += 1
                continue

            # --- HYBRID: Rule-based (primary) + ML probability (secondary) ---
            raw_pred, confidence, decision_src = hybrid_classify(
                features, classifier, x_vec
            )

            # --- Debug logging (first 50 frames + every 100th) ---
            if frame_idx < 50 or frame_idx % 100 == 0:
                ba_val = features.get('back_angle', 0)
                ka_val = features.get('knee_angle', 0)
                na_val = features.get('neck_angle', 0)
                print(f"[DEBUG] frame={frame_idx}  back={ba_val:.1f}°  knee={ka_val:.1f}°  "
                      f"neck={na_val:.1f}°  "
                      f"pred={LABEL_NAMES.get(raw_pred, '?')}  conf={confidence:.2f}  src={decision_src}")

            # --- Temporal smoothing (majority vote) ---
            recent_preds.append(raw_pred)
            counter = Counter(recent_preds)
            final_pred = counter.most_common(1)[0][0]

            label = LABEL_NAMES.get(final_pred, "UNKNOWN")
            color_bgr = LABEL_COLORS_BGR.get(final_pred, (200, 200, 200))
            reasons = build_reasons(features, label)

            # --- Voice alert logic ---
            if final_pred == 2:  # UNSAFE
                consecutive_unsafe += 1
            else:
                consecutive_unsafe = 0

            current_time = time.time()
            should_alert = (
                final_pred == 2
                and confidence >= conf_threshold
                and consecutive_unsafe >= UNSAFE_TRIGGER_FRAMES
                and (current_time - last_alert_time) >= ALERT_COOLDOWN
            )
            if should_alert:
                alert_text = _contextual_alert_text(features)
                trigger_voice_alert(alert_text)
                last_alert_time = current_time
                alert_box.error(f"🔊 **Voice Alert:** {alert_text}")
            elif final_pred != 2:
                alert_box.empty()

            # --- Draw overlay on frame ---
            draw_status_overlay(frame, features, label, confidence, reasons, person_box, color_bgr)

            # --- Write & display frame ---
            writer.write(frame)
            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            # --- Update Streamlit status panel ---
            css_color = STATUS_COLORS.get(label, "#888888")
            status_badge.markdown(
                f"""
                <div style="
                    background:{css_color}22;
                    border:2px solid {css_color};
                    border-radius:12px;
                    padding:14px 20px;
                    text-align:center;
                ">
                    <span style="font-size:2rem;font-weight:800;color:{css_color}">
                        {label}
                    </span><br>
                    <span style="font-size:0.9rem;color:#ccc;">Posture Status</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            conf_pct = int(confidence * 100)
            bar_color = css_color
            conf_meter.markdown(
                f"""
                <div style="text-align:center;margin-top:8px;">
                    <div style="font-size:0.85rem;color:#aaa;margin-bottom:4px;">
                        Model Confidence
                    </div>
                    <div style="background:#333;border-radius:8px;overflow:hidden;height:22px;">
                        <div style="
                            width:{conf_pct}%;
                            background:{bar_color};
                            height:100%;
                            border-radius:8px;
                            transition:width 0.3s;
                        "></div>
                    </div>
                    <div style="font-size:1.2rem;font-weight:700;color:{bar_color};">
                        {conf_pct}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            reason_text = " · ".join(reasons)
            if label == "UNSAFE":
                reason_box.error(f"⚠️ {reason_text}")
            elif label == "MODERATE":
                reason_box.warning(f"⚡ {reason_text}")
            else:
                reason_box.success(f"✅ {reason_text}")

            # --- Angle history for graph ---
            angle_history["frame"].append(frame_idx)
            angle_history["back_angle"].append(features.get("back_angle", float("nan")))
            angle_history["knee_angle"].append(features.get("knee_angle", float("nan")))
            angle_history["neck_angle"].append(features.get("neck_angle", float("nan")))

            # Update angle graph every 10 frames to reduce flicker
            if frame_idx % 10 == 0:
                _update_angle_chart(angle_chart, angle_history)

            _update_progress(progress, status_txt, frame_idx, total_frames, label)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        kill_all_tts()

    progress.progress(1.0)
    status_txt.success(f"✅ Processing complete — {frame_idx} frames analysed.")

    return angle_history


# ---------------------------------------------------------------------------
# Helper Rendering Functions
# ---------------------------------------------------------------------------

def _write_no_pose(
    frame: np.ndarray,
    writer,
    placeholder,
    msg: str,
) -> None:
    """Write a 'no pose' frame to disk and display it."""
    cv2.putText(
        frame, msg, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA,
    )
    writer.write(frame)
    placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True,
    )


def _update_progress(progress, status_txt, frame_idx: int, total: int, msg: str) -> None:
    progress.progress(min(frame_idx / max(total, 1), 1.0))
    status_txt.info(f"Frame {frame_idx}/{total} — {msg}")


def _update_angle_chart(placeholder, history: Dict[str, List[float]]) -> None:
    """Render a Plotly line chart of angle history inside a Streamlit placeholder."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        color_map = {
            "back_angle": "#ef5350",
            "knee_angle": "#42a5f5",
            "neck_angle": "#66bb6a",
        }
        label_map = {
            "back_angle": "Back Angle (deg)",
            "knee_angle": "Knee Angle (deg)",
            "neck_angle": "Neck Angle (deg)",
        }

        frames = history["frame"]
        for key in ["back_angle", "knee_angle", "neck_angle"]:
            vals = history[key]
            if vals:
                fig.add_trace(go.Scatter(
                    x=frames,
                    y=vals,
                    mode="lines",
                    name=label_map[key],
                    line=dict(color=color_map[key], width=2),
                ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,30,30,0.8)",
            font=dict(color="#cccccc", size=11),
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(title="Frame", gridcolor="#444"),
            yaxis=dict(title="Angle (°)", gridcolor="#444", range=[0, 185]),
        )

        placeholder.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Plotly not installed — show a simple table fallback
        import pandas as pd
        n = min(20, len(history["frame"]))
        df = pd.DataFrame({
            "Frame": history["frame"][-n:],
            "Back°": [round(v, 1) for v in history["back_angle"][-n:]],
            "Knee°": [round(v, 1) for v in history["knee_angle"][-n:]],
            "Neck°": [round(v, 1) for v in history["neck_angle"][-n:]],
        })
        placeholder.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Streamlit UI Layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Worker Pose Safety Monitor",
        page_icon="🦺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Global style ---
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f0f14;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #1a1a24;
        border-right: 1px solid #2a2a3a;
    }
    h1 { font-size: 1.8rem !important; color: #e8eaf6 !important; }
    .stProgress > div > div { border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#1a237e,#283593);
        border-radius:14px;
        padding:20px 28px;
        margin-bottom:20px;
        border-left:5px solid #7986cb;
    ">
        <h1 style="margin:0;color:#e8eaf6;">
            🦺 Worker Pose Safety Monitoring System
        </h1>
        <p style="margin:6px 0 0;color:#9fa8da;font-size:0.95rem;">
            YOLOv8s-pose → XGBoost Classifier · Real-time Ergonomic Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar configuration ---
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        yolo_choice = st.selectbox(
            "YOLOv8 Pose Model",
            ["yolov8s-pose.pt", "yolov8n-pose.pt"],
            index=0,
            help="yolov8s = more accurate (recommended); yolov8n = faster",
        )
        smoothing_window = st.slider(
            "Temporal Smoothing Window (frames)",
            min_value=3,
            max_value=15,
            value=SMOOTHING_WINDOW,
            help="Majority vote over last N frames — higher = more stable, less reactive",
        )
        conf_threshold = st.slider(
            "Voice Alert Confidence Threshold",
            min_value=0.50,
            max_value=0.99,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum model confidence required to trigger a voice alert",
        )
        st.markdown("---")
        st.markdown("**Feature Vector** (15 features)")
        st.caption("\n".join(f"• {f}" for f in FEATURE_COLS))
        st.markdown("---")
        st.markdown("""
        **Pipeline Steps**
        1. Upload MP4 video
        2. YOLOv8s detects poses
        3. Features extracted via `utils.py`
        4. XGBoost classifies posture
        5. Temporal smoothing applied
        6. Overlay drawn on each frame
        """)

    # --- Load model ---
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.pkl"

    if not model_path.exists():
        st.error(
            f"❌ `model.pkl` not found at `{model_path}`.\n\n"
            "Run the full pipeline first:\n"
            "```\npython dataset_generator_yolo.py\n"
            "python auto_label.py\n"
            "python train_model.py\n```"
        )
        return

    classifier = load_classifier(str(model_path))
    yolo_model = load_yolo(yolo_choice)

    expected = getattr(classifier, "n_features_in_", len(FEATURE_COLS))
    if expected != len(FEATURE_COLS):
        st.warning(
            f"⚠️ Model expects {expected} features but pipeline provides {len(FEATURE_COLS)}. "
            "Retrain the model with the new pipeline."
        )

    # --- Upload & process ---
    uploaded = st.file_uploader(
        "📁 Upload a video file (.mp4)",
        type=["mp4"],
        help="Upload a workplace video to analyse posture in real time.",
    )

    if uploaded is None:
        st.info("☝️ Upload a video to begin analysis.")

        # Show instruction cards when idle
        c1, c2, c3 = st.columns(3)
        _info_card(c1, "🎯", "SAFE",
                   "Back > 150° · Knee 70°-120° (lifting) · Upright",
                   STATUS_COLORS["SAFE"])
        _info_card(c2, "⚡", "MODERATE",
                   "Back 120°-150° · Knee 120°-150° · Transition",
                   STATUS_COLORS["MODERATE"])
        _info_card(c3, "🚨", "UNSAFE",
                   "Back < 120° · Locked knees (>150°) + bent back · Neck < 120°",
                   STATUS_COLORS["UNSAFE"])
        return

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        temp_input = Path(tmp.name)

    output_video = base_dir / "output.mp4"

    st.markdown("### 🎬 Live Analysis")
    try:
        angle_history = process_video(
            temp_input,
            output_video,
            classifier,
            yolo_model,
            conf_threshold=conf_threshold,
            smoothing_window=smoothing_window,
        )
    except Exception as exc:
        st.error(f"❌ Processing failed: {exc}")
        kill_all_tts()
        return
    finally:
        try:
            temp_input.unlink()
        except Exception:
            pass

    # --- Final output ---
    st.markdown("---")
    st.markdown("### 📊 Analysis Complete")

    tab_video, tab_graph = st.tabs(["📹 Processed Video", "📈 Angle Trends"])

    with tab_video:
        if output_video.exists():
            st.video(str(output_video))
            with output_video.open("rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f,
                    file_name="pose_safety_output.mp4",
                    mime="video/mp4",
                )

    with tab_graph:
        if angle_history["frame"]:
            _update_angle_chart(st.empty(), angle_history)

            # Summary statistics table
            import pandas as pd
            stats = {}
            for key in ["back_angle", "knee_angle", "neck_angle"]:
                vals = [v for v in angle_history[key] if not math.isnan(v)]
                if vals:
                    stats[key] = {
                        "Mean°": round(float(np.mean(vals)), 1),
                        "Min°": round(float(np.min(vals)), 1),
                        "Max°": round(float(np.max(vals)), 1),
                        "Std°": round(float(np.std(vals)), 1),
                    }
            if stats:
                st.markdown("**Angle Summary Statistics**")
                st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
        else:
            st.info("No angle data recorded — check your video.")


# ---------------------------------------------------------------------------
# UI Helper: Info Card
# ---------------------------------------------------------------------------

def _info_card(col, icon: str, title: str, description: str, color: str) -> None:
    col.markdown(
        f"""
        <div style="
            background:{color}18;
            border:1.5px solid {color};
            border-radius:12px;
            padding:18px;
            text-align:center;
            height:140px;
        ">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:1.2rem;font-weight:700;color:{color};margin:4px 0;">
                {title}
            </div>
            <div style="font-size:0.78rem;color:#aaa;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
