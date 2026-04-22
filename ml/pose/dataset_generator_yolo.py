"""
dataset_generator_yolo.py — YOLOv8-Only Dataset Generator
==========================================================
Worker Pose Safety Monitoring System — Production Pipeline

Replaces the old MediaPipe-based dataset_generator.py entirely.

Pipeline:
  For each MP4 video in Videos/:
    1. Read every N-th frame (frame_step=2)
    2. Apply random data augmentation (brightness, blur)
    3. Run YOLOv8s-pose inference
    4. Select the primary person (largest bbox, closest to center)
    5. Validate keypoint confidence on essential joints
    6. Extract all features via utils.extract_all_features()
       → angles, temporal features, normalized coordinates
    7. Write row to dataset.csv

Output CSV columns:
  video_name, frame_id,
  back_angle, knee_angle, neck_angle, elbow_angle,
  back_vel, knee_vel, neck_vel,
  back_acc, knee_acc,
  norm_shoulder_x, norm_shoulder_y,
  norm_hip_x, norm_hip_y,
  norm_knee_x, norm_knee_y,
  label  (empty — filled by auto_label.py)

Usage:
  python dataset_generator_yolo.py
"""

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    FEATURE_COLS,
    make_empty_buffers,
    select_primary_person,
    extract_all_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YOLO_MODEL_NAME: str = "yolov8s-pose.pt"   # Use yolov8s for higher accuracy

FRAME_STEP: int = 2          # Process every 2nd frame (speed vs coverage)
YOLO_CONF_THR: float = 0.25  # YOLO detection confidence threshold
FRAME_W: int = 640
FRAME_H: int = 480

# Seed for reproducible augmentation
random.seed(42)
np.random.seed(42)

# All CSV columns in order
CSV_COLUMNS: List[str] = ["video_name", "frame_id"] + FEATURE_COLS + ["label"]


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

def augment_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply randomized augmentation to a BGR frame before pose inference.

    Augmentations (applied independently):
      - Random brightness & contrast scaling (alpha 1.0–1.4, beta 0–25)
      - 50% chance of a random Gaussian blur (kernel 3×3 or 5×5)

    This improves dataset diversity without changing joint coordinates.
    """
    alpha = random.uniform(1.0, 1.4)
    beta = random.uniform(0, 25)
    out = cv2.convertScaleAbs(frame_bgr, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out


# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    yolo_model: YOLO,
    frame_step: int = FRAME_STEP,
) -> List[Dict[str, object]]:
    """
    Extract feature rows from a single video file.

    Args:
        video_path:  Path to the .mp4 file
        yolo_model:  Pre-loaded YOLOv8 pose model
        frame_step:  Sample every N-th frame

    Returns:
        List of row dicts, one per accepted frame.
    """
    rows: List[Dict[str, object]] = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path.name}")
        return rows

    # Fresh temporal buffers for each video (no bleed between clips)
    buffers = make_empty_buffers()

    frame_id = 0
    skipped_no_pose = 0
    skipped_low_conf = 0
    skipped_nan = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Skip frames not at the step boundary
        if frame_id % frame_step != 0:
            frame_id += 1
            continue

        # Resize to standard resolution
        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)

        # Apply augmentation BEFORE pose inference for diverse training data
        augmented = augment_frame(frame)

        # Run YOLOv8 pose inference
        results = yolo_model.predict(
            augmented,
            verbose=False,
            conf=YOLO_CONF_THR,
            imgsz=FRAME_W,
        )
        result = results[0]

        # Skip if no detections
        if (
            result.keypoints is None
            or result.boxes is None
            or len(result.boxes) == 0
        ):
            skipped_no_pose += 1
            frame_id += 1
            continue

        # Extract arrays
        boxes = result.boxes.xyxy.cpu().numpy()          # (N, 4)
        kps_xy_all = result.keypoints.xy.cpu().numpy()   # (N, 17, 2)
        kps_conf_all = result.keypoints.conf.cpu().numpy()  # (N, 17)

        # Select primary person
        person_idx = select_primary_person(boxes, FRAME_W, FRAME_H)
        if person_idx is None:
            skipped_no_pose += 1
            frame_id += 1
            continue

        kps_xy = kps_xy_all[person_idx]
        kps_conf = kps_conf_all[person_idx]

        # Extract features (includes confidence validation inside)
        feats = extract_all_features(kps_xy, kps_conf, buffers)

        if feats is None:
            skipped_low_conf += 1
            frame_id += 1
            continue

        # Check for any NaN in extracted features (degenerate geometry)
        if any(math.isnan(v) for v in feats.values()):
            skipped_nan += 1
            frame_id += 1
            continue

        # Build row
        row: Dict[str, object] = {
            "video_name": video_path.name,
            "frame_id": frame_id,
            "label": "",  # to be filled by auto_label.py
        }
        row.update(feats)
        rows.append(row)

        frame_id += 1

    cap.release()

    print(
        f"  [OK] {video_path.name}: {len(rows)} rows | "
        f"skipped no_pose={skipped_no_pose}, low_conf={skipped_low_conf}, nan={skipped_nan}"
    )
    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_dataset_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    """Write all extracted feature rows to a CSV file."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            # Ensure all columns present; fill missing with empty string
            clean = {col: row.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(clean)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    video_dir = base_dir / "Videos"
    output_csv = base_dir / "dataset.csv"

    # Accept both 'Videos' and 'videos' folder names
    if not video_dir.exists():
        video_dir = base_dir / "videos"

    video_files = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []

    if not video_files:
        print(f"[INFO] No .mp4 files found in: {video_dir}")
        write_dataset_csv(output_csv, [])
        print(f"[INFO] Empty dataset created at: {output_csv}")
        return

    print(f"[INFO] YOLOv8 Model  : {YOLO_MODEL_NAME}")
    print(f"[INFO] Frame step    : {FRAME_STEP} (every {FRAME_STEP}nd frame)")
    print(f"[INFO] Videos found  : {len(video_files)}")
    print(f"[INFO] Output CSV    : {output_csv}")
    print(f"[INFO] Features      : {FEATURE_COLS}")
    print()

    # Load YOLOv8 pose model once (reused across all videos)
    print(f"[INFO] Loading {YOLO_MODEL_NAME} ...")
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print(f"[INFO] Model loaded. Starting extraction...\n")

    all_rows: List[Dict[str, object]] = []

    for video_path in video_files:
        print(f"[INFO] Processing: {video_path.name}")
        rows = process_video(video_path, yolo_model, frame_step=FRAME_STEP)
        all_rows.extend(rows)

    write_dataset_csv(output_csv, all_rows)

    print(f"\n{'='*60}")
    print(f"[DONE] Dataset saved : {output_csv}")
    print(f"[DONE] Total rows    : {len(all_rows)}")
    print(f"[DONE] Columns       : {CSV_COLUMNS}")
    print(f"\nNext step: run  python auto_label.py")


if __name__ == "__main__":
    main()
