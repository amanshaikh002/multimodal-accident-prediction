"""
generate_architecture_diagram.py
================================
Renders a high-resolution PNG of the system architecture for the
Multimodal Vision Audio Framework. Uses only matplotlib so no external
rendering binaries (Chromium / Graphviz) are required.

Run from the project root:
    python scripts/generate_architecture_diagram.py

Output:
    docs/system_architecture.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG          = "#0b1220"      # canvas background
PANEL_BG    = "#1e293b"      # large layer cards
PANEL_EDGE  = "#334155"

CLIENT_FILL = "#1e3a8a"      # blue
CLIENT_EDGE = "#3b82f6"

ROUTE_FILL  = "#312e81"      # indigo
ROUTE_EDGE  = "#6366f1"

SVC_FILL    = "#064e3b"      # green
SVC_EDGE    = "#10b981"

UTIL_FILL   = "#134e4a"      # teal
UTIL_EDGE   = "#14b8a6"

MODEL_FILL  = "#3f0e0e"      # dark red
MODEL_EDGE  = "#dc2626"

STORE_FILL  = "#451a03"      # amber
STORE_EDGE  = "#f59e0b"

DECISION_FILL = "#3b0764"    # purple
DECISION_EDGE = "#a855f7"

ARROW_DATA  = "#cbd5e1"      # light slate for data flow
ARROW_OPT   = "#64748b"      # dimmer for optional/fallback
TEXT_BRIGHT = "#f8fafc"
TEXT_DIM    = "#cbd5e1"

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(28, 20), dpi=170)
ax.set_xlim(0, 200)
ax.set_ylim(0, 140)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def panel(x, y, w, h, title, fill, edge, title_color=TEXT_BRIGHT):
    """A large translucent backdrop for a layer."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.4, edgecolor=edge, facecolor=fill, alpha=0.35,
    )
    ax.add_patch(rect)
    ax.text(x + 1.2, y + h - 1.3, title,
            color=title_color, fontsize=14, fontweight="bold", va="top",
            family="DejaVu Sans")


def box(x, y, w, h, lines, fill, edge, text_color=TEXT_BRIGHT,
        fontsize=10.5, bold_first=True, alpha=0.95):
    """A single component box. `lines` is a list of strings."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.7",
        linewidth=1.4, edgecolor=edge, facecolor=fill, alpha=alpha,
    )
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    line_h = h / (n + 1)
    for i, line in enumerate(lines):
        weight = "bold" if (bold_first and i == 0) else "normal"
        size   = fontsize if i == 0 else fontsize - 1
        ax.text(x + w / 2, y + h - line_h * (i + 1),
                line, ha="center", va="center",
                color=text_color, fontsize=size, fontweight=weight,
                family="DejaVu Sans")


def cyl(x, y, w, h, lines, fill, edge, text_color=TEXT_BRIGHT,
        fontsize=10):
    """Storage cylinder shape (rounded top + bottom)."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=2.0",
        linewidth=1.4, edgecolor=edge, facecolor=fill, alpha=0.95,
    )
    ax.add_patch(rect)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    line_h = h / (n + 1)
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        size   = fontsize if i == 0 else fontsize - 1
        ax.text(x + w / 2, y + h - line_h * (i + 1),
                line, ha="center", va="center",
                color=text_color, fontsize=size, fontweight=weight,
                family="DejaVu Sans")


def arrow(x1, y1, x2, y2, color=ARROW_DATA, lw=1.6, style="->", alpha=0.85,
          connection="arc3,rad=0"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, alpha=alpha,
        connectionstyle=connection,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def label(x, y, text, color=TEXT_DIM, fontsize=9.5, ha="center"):
    ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha,
            family="DejaVu Sans", style="italic")


# ---------------------------------------------------------------------------
# 1. TITLE
# ---------------------------------------------------------------------------

ax.text(100, 135.5,
        "Multimodal Vision Audio Framework  —  System Architecture",
        ha="center", va="center", color=TEXT_BRIGHT,
        fontsize=22, fontweight="bold", family="DejaVu Sans")
ax.text(100, 132.2,
        "PPE compliance · Pose ergonomics · Accident detection · Fire hazard · Anomaly sound",
        ha="center", va="center", color=TEXT_DIM,
        fontsize=12.5, style="italic", family="DejaVu Sans")


# ---------------------------------------------------------------------------
# 2. CLIENT LAYER (top)
# ---------------------------------------------------------------------------

panel(4, 113, 192, 16, "1.  CLIENT LAYER  —  React 19 + Vite", PANEL_BG, PANEL_EDGE)

box(7,   116, 24, 9, ["App.jsx", "top-level state,", "module selector,", "Analyse button"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(33,  116, 24, 9, ["Per-mode Panels", "PPE / Pose / Combined", "Fire / Sound / All"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(59,  116, 24, 9, ["VideoPanel", "HTML5 <video>", "/output/*.mp4"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(85,  116, 24, 9, ["ViolationsList", "Recommendations", "frame-by-frame"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(111, 116, 24, 9, ["useAudioAlerts", "SpeechSynthesis", "voice alerts"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(137, 116, 24, 9, ["suggestionMap.js", "enrich PPE/Pose/", "Combined/Fire/Sound"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=10)
box(163, 116, 30, 9, ["AllPanel (dynamic)", "CRITICAL banner switches between",
                      "FIRE HAZARD / WORKER ACCIDENT / both"],
    CLIENT_FILL, CLIENT_EDGE, fontsize=9.5)


# ---------------------------------------------------------------------------
# 3. CLIENT <-> BACKEND ARROWS
# ---------------------------------------------------------------------------

# Outgoing request
arrow(50, 113, 50, 109, color="#60a5fa", lw=2.0)
ax.text(51.5, 110.8, "POST /detect?mode=<...>",
        color="#60a5fa", fontsize=10, fontweight="bold", family="DejaVu Sans")

# Incoming JSON
arrow(78, 109, 78, 113, color="#86efac", lw=2.0)
ax.text(79.5, 110.8, "JSON: status, violations, events",
        color="#86efac", fontsize=10, fontweight="bold", family="DejaVu Sans")

# Static video
arrow(150, 113, 150, 109, color="#60a5fa", lw=2.0)
ax.text(151.5, 110.8, "GET /output/*.mp4 (HTTP 206)",
        color="#60a5fa", fontsize=10, fontweight="bold", family="DejaVu Sans")


# ---------------------------------------------------------------------------
# 4. BACKEND ROUTES + SERVICES
# ---------------------------------------------------------------------------

panel(4, 56, 192, 52, "2.  FASTAPI BACKEND  —  Uvicorn", PANEL_BG, PANEL_EDGE)

# Routes row
ax.text(8, 105, "Routes", color=TEXT_DIM, fontsize=11.5,
        fontweight="bold", family="DejaVu Sans", style="italic")
box(20,  100, 60, 7, ["routes/detect.py — UNIFIED",
                       "POST /detect?mode=<ppe | pose | fire | sound | combined | all>"],
    ROUTE_FILL, ROUTE_EDGE, fontsize=10)
box(85,  100, 22, 7, ["Legacy routes",
                       "/detect/{ppe,pose,fire,sound}"],
    ROUTE_FILL, ROUTE_EDGE, fontsize=9.5)
box(112, 100, 24, 7, ["main.py", "CORS / health / static mount"],
    ROUTE_FILL, ROUTE_EDGE, fontsize=9.5)
box(141, 100, 51, 7, ["StaticFiles  →  /output/*",
                       "serves all *_annotated.mp4 files"],
    ROUTE_FILL, ROUTE_EDGE, fontsize=10)

# Service row
ax.text(8, 90, "Services", color=TEXT_DIM, fontsize=11.5,
        fontweight="bold", family="DejaVu Sans", style="italic")

# ── PPE service ─────────────────────────────────
box(8, 76, 30, 12,
    ["ppe_service",
     "stride=2 · YOLOv8 PPE",
     "+ tri-state SAFE/UNSAFE/UNKNOWN",
     "+ sticky helmet/vest (1.5 s)",
     "+ YOLO-World hazard pass"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# ── Pose service ────────────────────────────────
box(40, 76, 32, 12,
    ["pose_service",
     "YOLOv8s-pose (17 kps)",
     "+ PoseTracker (IoU)",
     "+ AccidentDetector (5 events)",
     "+ ergonomic ML classifier"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# ── Fire service ────────────────────────────────
box(74, 76, 30, 12,
    ["fire_service",
     "HF auto-download YOLO",
     "imgsz=1280 · aspect preserved",
     "2-frame persistence",
     "draw_banner kwarg for chaining"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# ── Sound service ───────────────────────────────
box(106, 76, 30, 12,
    ["sound_service",
     "librosa + ffmpeg fallback",
     "3 s window / 1 s hop · MFCC",
     "RandomForest + 2-win persistence",
     "graceful 200 on no-audio"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# ── Combined service ────────────────────────────
box(138, 76, 26, 12,
    ["combined_service",
     "PPE + Pose synchronous",
     "shared frame, one writer",
     "sticky-PPE active here too",
     "merge_results + decision"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# ── All (full platform) chain ────────────────────
box(166, 76, 26, 12,
    ["all  (Full Platform)",
     "1. combined → mp4",
     "2. fire (draw_banner=False)",
     "    overlays on top",
     "→ all_annotated.mp4"],
    SVC_FILL, SVC_EDGE, fontsize=9.2)

# Route → Service arrows
arrow(50, 100, 23, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)
arrow(50, 100, 56, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)
arrow(50, 100, 89, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)
arrow(50, 100, 121, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)
arrow(50, 100, 151, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)
arrow(50, 100, 179, 88, color=ARROW_DATA, lw=1.2, alpha=0.6)

# Combined depends on PPE+Pose (dotted teal)
arrow(151, 76, 56, 76, color="#5eead4", lw=1.4, style="->", alpha=0.55,
      connection="arc3,rad=-0.18")
arrow(151, 76, 25, 76, color="#5eead4", lw=1.4, style="->", alpha=0.55,
      connection="arc3,rad=-0.22")

# All chain: combined -> fire
arrow(179, 76, 89, 76, color="#fde68a", lw=1.5, style="->", alpha=0.7,
      connection="arc3,rad=-0.18")

# Utils row
ax.text(8, 71, "Utilities (shared helpers)", color=TEXT_DIM, fontsize=11.5,
        fontweight="bold", family="DejaVu Sans", style="italic")

box(8,  60, 26, 9, ["ppe_utils",
                    "tri-state evaluator,",
                    "motion score,",
                    "hazard override"],
    UTIL_FILL, UTIL_EDGE, fontsize=9)
box(36, 60, 24, 9, ["pose_utils",
                    "vector angles,",
                    "feature vector,",
                    "hybrid_classify"],
    UTIL_FILL, UTIL_EDGE, fontsize=9)
box(62, 60, 24, 9, ["pose_tracker",
                    "greedy IoU",
                    "stable track_id",
                    "rolling history"],
    UTIL_FILL, UTIL_EDGE, fontsize=9)
box(88, 60, 30, 9, ["accident_detector",
                    "FALL · STRUCK · CRUSHED",
                    "MOTIONLESS_DOWN · STUMBLE",
                    "+ TTL OverlayRenderer"],
    UTIL_FILL, UTIL_EDGE, fontsize=9)
box(120, 60, 22, 9, ["video_utils",
                     "open_video,",
                     "writer + H.264",
                     "re-encode"],
    UTIL_FILL, UTIL_EDGE, fontsize=9)
box(144, 60, 48, 9, ["combined_service decision engine",
                     "get_final_status(ppe, pose, fire, accident, sound)",
                     "→ CRITICAL · HIGH RISK · UNSAFE · MODERATE · SAFE"],
    UTIL_FILL, UTIL_EDGE, fontsize=9.2)

# Service -> Util arrows
arrow(20, 76, 21, 69, color=ARROW_DATA, lw=1.0, alpha=0.5)        # ppe -> ppe_utils
arrow(56, 76, 48, 69, color=ARROW_DATA, lw=1.0, alpha=0.5)        # pose -> pose_utils
arrow(56, 76, 74, 69, color=ARROW_DATA, lw=1.0, alpha=0.5)        # pose -> tracker
arrow(56, 76, 103, 69, color=ARROW_DATA, lw=1.0, alpha=0.5)       # pose -> accident
arrow(151, 76, 21, 69, color=ARROW_DATA, lw=1.0, alpha=0.4,
      connection="arc3,rad=-0.32")                                # combined -> ppe_utils
arrow(151, 76, 48, 69, color=ARROW_DATA, lw=1.0, alpha=0.4,
      connection="arc3,rad=-0.20")                                # combined -> pose_utils


# ---------------------------------------------------------------------------
# 5. STORAGE / MODELS LAYER
# ---------------------------------------------------------------------------

panel(4, 24, 192, 30, "3.  MODELS & STORAGE", PANEL_BG, PANEL_EDGE)

# Models row
cyl(8,  39, 28, 11,
    ["ppe_model.pt",
     "custom YOLOv8",
     "human · helmet · vest ·",
     "gloves · boots"],
    MODEL_FILL, MODEL_EDGE, fontsize=9.2)
cyl(38, 39, 28, 11,
    ["yolov8s-pose.pt",
     "17 COCO keypoints",
     "auto-downloaded by",
     "ultralytics"],
    MODEL_FILL, MODEL_EDGE, fontsize=9.2)
cyl(68, 39, 28, 11,
    ["model.pkl",
     "ergonomic classifier",
     "RandomForest, 9 features",
     "trained via SMOTE + video split"],
    MODEL_FILL, MODEL_EDGE, fontsize=9)
cyl(98, 39, 30, 11,
    ["HuggingFace cache",
     "SalahALHaismawi /",
     "yolov26-fire-detection",
     "{fire, other, smoke}"],
    MODEL_FILL, MODEL_EDGE, fontsize=9)
cyl(130, 39, 28, 11,
    ["yolov8s-worldv2.pt",
     "open-vocab YOLO-World",
     "PPE hazard prompts +",
     "fire fallback"],
    MODEL_FILL, MODEL_EDGE, fontsize=9)
cyl(160, 39, 32, 11,
    ["audio_model.pkl",
     "Sound Anomaly /",
     "RandomForest binary",
     "(normal vs anomaly)"],
    MODEL_FILL, MODEL_EDGE, fontsize=9)

# Service -> Model arrows
arrow(23, 76, 22, 50, color=MODEL_EDGE, lw=1.2, alpha=0.55)        # ppe -> ppe_model
arrow(56, 76, 52, 50, color=MODEL_EDGE, lw=1.2, alpha=0.55)        # pose -> yolov8s-pose
arrow(56, 76, 82, 50, color=MODEL_EDGE, lw=1.2, alpha=0.55)        # pose -> model.pkl
arrow(89, 76, 113, 50, color=MODEL_EDGE, lw=1.2, alpha=0.55)       # fire -> HF cache
arrow(23, 76, 144, 50, color=MODEL_EDGE, lw=0.9, alpha=0.4,
      connection="arc3,rad=0.10")                                  # ppe -> world (hazards)
arrow(89, 76, 144, 50, color=MODEL_EDGE, lw=0.9, alpha=0.35,
      connection="arc3,rad=-0.10")                                 # fire -> world (fallback)
arrow(121, 76, 176, 50, color=MODEL_EDGE, lw=1.2, alpha=0.55)      # sound -> audio_model

# Storage row (annotated outputs)
cyl(8,  26, 184, 11,
    ["temp/output/   —   annotated videos served via /output/*",
     "ppe_annotated.mp4   ·   pose_annotated.mp4   ·   fire_annotated.mp4   ·   "
     "sound_annotated.mp4   ·   combined_annotated.mp4   ·   all_annotated.mp4"],
    STORE_FILL, STORE_EDGE, fontsize=10.5)


# ---------------------------------------------------------------------------
# 6. DECISION ENGINE (right side floating)
# ---------------------------------------------------------------------------

panel(150, 4, 46, 18, "4.  UNIFIED DECISION ENGINE", PANEL_BG, PANEL_EDGE)

box(152, 6, 42, 14,
    ["Priority (highest first):",
     "",
     "CRITICAL    fire UNSAFE  OR  accident CRITICAL",
     "HIGH RISK   PPE + Pose both UNSAFE",
     "UNSAFE      any module UNSAFE / accident WARN",
     "MODERATE    Pose MODERATE only",
     "SAFE        all clear"],
    DECISION_FILL, DECISION_EDGE,
    fontsize=9.4, bold_first=True)


# ---------------------------------------------------------------------------
# 7. FULL-PLATFORM CHAIN illustrative box (left of decision)
# ---------------------------------------------------------------------------

panel(4, 4, 144, 18, "5.  FULL PLATFORM PIPELINE  (mode = all)", PANEL_BG, PANEL_EDGE)

# Stage boxes
box(8,  9, 32, 10,
    ["Stage 1 · combined_service",
     "PPE + Pose YOLO synchronous",
     "skeleton + per-person box",
     "+ accident overlays + banner"],
    SVC_FILL, SVC_EDGE, fontsize=9)

box(54, 9, 32, 10,
    ["Stage 2 · fire_service",
     "HF fire YOLO @ imgsz=1280",
     "draw_banner=False",
     "→ overlays fire bbox only"],
    SVC_FILL, SVC_EDGE, fontsize=9)

box(100, 9, 44, 10,
    ["all_annotated.mp4",
     "single unified video with",
     "PPE · Pose · Accident · Fire",
     "served via /output/all_annotated.mp4"],
    STORE_FILL, STORE_EDGE, fontsize=9)

arrow(40, 14, 54, 14, color="#fde68a", lw=2.2)
ax.text(47, 16, "combined_annotated.mp4", color="#fde68a",
        fontsize=8.5, ha="center", family="DejaVu Sans", fontweight="bold")
arrow(86, 14, 100, 14, color="#fde68a", lw=2.2)


# ---------------------------------------------------------------------------
# 8. LEGEND
# ---------------------------------------------------------------------------

# Small color-key strip on the title row right side
def legend_swatch(x, y, color, edge, label_text):
    rect = FancyBboxPatch((x, y), 2.2, 1.5,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          linewidth=1.0, edgecolor=edge, facecolor=color, alpha=0.95)
    ax.add_patch(rect)
    ax.text(x + 2.6, y + 0.75, label_text, color=TEXT_DIM,
            fontsize=8.5, va="center", family="DejaVu Sans")


legend_swatch(122, 130, CLIENT_FILL,   CLIENT_EDGE,   "Client (React)")
legend_swatch(140, 130, ROUTE_FILL,    ROUTE_EDGE,    "Routes")
legend_swatch(155, 130, SVC_FILL,      SVC_EDGE,      "Services")
legend_swatch(170, 130, UTIL_FILL,     UTIL_EDGE,     "Utilities")
legend_swatch(185, 130, MODEL_FILL,    MODEL_EDGE,    "Models")

legend_swatch(122, 127.5, STORE_FILL,    STORE_EDGE,    "Annotated outputs")
legend_swatch(155, 127.5, DECISION_FILL, DECISION_EDGE, "Decision engine")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR      = PROJECT_ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = OUT_DIR / "system_architecture.png"

plt.savefig(
    OUT_PATH,
    dpi=200,
    facecolor=BG,
    edgecolor="none",
    bbox_inches="tight",
    pad_inches=0.4,
)
plt.close(fig)

size_kb = OUT_PATH.stat().st_size / 1024
print(f"OK  wrote: {OUT_PATH}  ({size_kb:.1f} KB)")
