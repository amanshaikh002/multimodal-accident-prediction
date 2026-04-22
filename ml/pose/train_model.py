"""
train_model.py — XGBoost Posture Classifier Training
=====================================================
Worker Pose Safety Monitoring System — Production Pipeline

Trains an XGBoost (or LightGBM fallback) classifier on the 15-feature
vector produced by the YOLOv8 pipeline.

Pipeline:
  1. Load labeled_dataset.csv
  2. Validate & clean features
  3. Video-based train/test split (no data leakage between clips)
  4. SMOTE oversampling on the TRAIN set to handle class imbalance
  5. Add Gaussian noise to angle features (robustness)
  6. Train XGBoost with class weighting
  7. Evaluate: accuracy, confusion matrix, classification report
  8. Plot & print feature importances
  9. Save model.pkl

Requirements:
  pip install xgboost imbalanced-learn

Feature vector (15 features — matches utils.FEATURE_COLS):
  back_angle, knee_angle, neck_angle, elbow_angle,
  back_vel, knee_vel, neck_vel,
  back_acc, knee_acc,
  norm_shoulder_x, norm_shoulder_y,
  norm_hip_x, norm_hip_y,
  norm_knee_x, norm_knee_y

Usage:
  python train_model.py
"""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils import FEATURE_COLS

# Fix Windows console encoding (handles emoji in filenames like '👷')
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback — emoji will be replaced by '?'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANGLE_NOISE_STD: float = 2.5   # Gaussian noise std (degrees) on angle features
ANGLE_COLS_COUNT: int = 4       # First N features are angles that get noise
TEST_FRACTION: float = 0.20     # 20% of videos held out for testing
RANDOM_STATE: int = 42
LABEL_MAP: dict = {"safe": 0, "moderate": 1, "unsafe": 2}
LABEL_NAMES: dict = {0: "safe", 1: "moderate", 2: "unsafe"}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def video_based_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by VIDEO SOURCE so that no frames from the same video
    appear in both the train and test sets. This prevents data leakage caused
    by temporal correlation between consecutive frames.

    Args:
        df: full labeled dataframe (must contain 'video_name' column)
        test_fraction: fraction of unique videos to hold out
        random_state: RNG seed for reproducibility

    Returns:
        (train_df, test_df)
    """
    video_names = df["video_name"].unique()
    n_test = max(1, int(len(video_names) * test_fraction))

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(video_names)

    test_videos = set(shuffled[:n_test])
    train_videos = set(shuffled[n_test:])

    train_df = df[df["video_name"].isin(train_videos)].copy()
    test_df = df[df["video_name"].isin(test_videos)].copy()

    def _safe_names(names) -> str:
        """Encode names as ASCII for safe terminal printing on Windows."""
        return str(sorted(str(n).encode('ascii', errors='replace').decode('ascii')
                          for n in names))

    print(f"\n--- Video-Based Split ---")
    print(f"  Total videos : {len(video_names)}")
    print(f"  Train ({len(train_videos):2d}) : {_safe_names(train_videos)}")
    print(f"  Test  ({len(test_videos):2d}) : {_safe_names(test_videos)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test  samples: {len(test_df)}")

    return train_df, test_df


def add_angle_noise(X: np.ndarray, n_angle_cols: int, std: float) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to the first `n_angle_cols` features (angles).
    Coordinate and temporal features are left unchanged.
    Angles are clamped to [0, 180] after adding noise.
    """
    X_noisy = X.copy()
    noise = np.random.normal(0.0, std, size=(X.shape[0], n_angle_cols))
    X_noisy[:, :n_angle_cols] += noise
    X_noisy[:, :n_angle_cols] = np.clip(X_noisy[:, :n_angle_cols], 0.0, 180.0)
    return X_noisy


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the training set to address class imbalance.
    Requires the imbalanced-learn package.

    Falls back to light manual upsampling if imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, _min_class_count(y) - 1))
        X_res, y_res = sm.fit_resample(X, y)
        print(f"[INFO] SMOTE applied. New training size: {len(X_res)}")
        return X_res, y_res
    except ImportError:
        print("[WARN] imbalanced-learn not installed. Using manual upsampling.")
        return _manual_upsample(X, y)
    except Exception as e:
        print(f"[WARN] SMOTE failed ({e}). Using manual upsampling.")
        return _manual_upsample(X, y)


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _manual_upsample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Light upsampling: bring all classes to the majority count."""
    from sklearn.utils import resample
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    X_parts, y_parts = [], []
    for cls in classes:
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        if len(X_cls) < max_count:
            X_cls, y_cls = resample(X_cls, y_cls, n_samples=max_count, random_state=RANDOM_STATE)
        X_parts.append(X_cls)
        y_parts.append(y_cls)
    return np.vstack(X_parts), np.concatenate(y_parts)


def load_xgboost():
    """Load XGBoost classifier; fall back to LightGBM if unavailable."""
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        print("[INFO] Using XGBoost classifier")
        return model
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        print("[INFO] Using LightGBM classifier (XGBoost not found)")
        return model
    except ImportError:
        pass

    # Final fallback: Random Forest (always available)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("[WARN] XGBoost/LightGBM not installed. Using RandomForest fallback.")
    print("       Install:  pip install xgboost")
    return model


def print_feature_importances(model, feature_names: list) -> None:
    """Print feature importances as a bar chart in the terminal."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = list(zip(feature_names, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)

    print("\nFeature Importances (sorted):")
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:22s}: {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main Training Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "labeled_dataset.csv"
    model_path = base_dir / "model.pkl"

    # --- 1. Load data ---
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {input_csv}\n"
            "Run auto_label.py first."
        )

    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} rows from {input_csv.name}")

    # --- 2. Validate columns ---
    required = set(FEATURE_COLS) | {"label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled_dataset.csv: {sorted(missing)}")

    # Convert to numeric, drop NaN
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=FEATURE_COLS + ["label"])
    df = df[df["label"].isin(LABEL_MAP)].copy()

    if df.empty:
        raise ValueError("No valid rows after cleaning. Check your labeled_dataset.csv.")

    df["label_encoded"] = df["label"].map(LABEL_MAP)

    print(f"[INFO] Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    print(f"[INFO] Total clean samples: {len(df)}")

    # Class distribution
    print("\n--- Class Distribution (raw) ---")
    for cls_id in [0, 1, 2]:
        count = int((df["label_encoded"] == cls_id).sum())
        pct = 100.0 * count / len(df)
        print(f"  {LABEL_NAMES[cls_id]:8s}: {count:5d}  ({pct:.1f}%)")

    # --- 3. Video-based split ---
    if "video_name" in df.columns and df["video_name"].nunique() > 1:
        train_df, test_df = video_based_split(df)

        # Fallback to random split if test set too small or mono-class
        if len(test_df) < 15 or test_df["label_encoded"].nunique() < 2:
            print("[WARN] Video split insufficient — falling back to random split.")
            X_all = df[FEATURE_COLS].values
            y_all = df["label_encoded"].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_FRACTION,
                random_state=RANDOM_STATE, stratify=y_all,
            )
        else:
            X_train = train_df[FEATURE_COLS].values
            y_train = train_df["label_encoded"].values
            X_test = test_df[FEATURE_COLS].values
            y_test = test_df["label_encoded"].values
    else:
        X_all = df[FEATURE_COLS].values
        y_all = df["label_encoded"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=TEST_FRACTION,
            random_state=RANDOM_STATE, stratify=y_all,
        )

    # --- 4. SMOTE oversampling on TRAIN ONLY ---
    X_train, y_train = apply_smote(X_train, y_train)

    # --- 5. Add Gaussian noise to angle features (TRAIN only) ---
    X_train = add_angle_noise(X_train, n_angle_cols=ANGLE_COLS_COUNT, std=ANGLE_NOISE_STD)

    print(f"\n[INFO] Train samples (after SMOTE + noise): {len(X_train)}")
    print(f"[INFO] Test  samples (clean, no noise)     : {len(X_test)}")

    # --- 6. Train model ---
    model = load_xgboost()
    print(f"\n[INFO] Training...")
    model.fit(X_train, y_train)
    print(f"[INFO] Training complete.")

    # --- 7. Evaluate ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    present = sorted(set(y_test) | set(y_pred))
    target_names = [LABEL_NAMES[c] for c in present]
    report = classification_report(y_test, y_pred, labels=present, target_names=target_names)

    print(f"\n{'='*55}")
    print(f"Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"{'='*55}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join(f"{LABEL_NAMES[c]:8s}" for c in present)
    print(header)
    for i, cls in enumerate(present):
        row_label = f"{LABEL_NAMES[cls]:8s}"
        row_vals = "  ".join(f"{cm[i][j]:8d}" for j in range(len(present)))
        print(f"  {row_label} {row_vals}")
    print("\nClassification Report:")
    print(report)

    # --- 8. Feature importances ---
    print_feature_importances(model, FEATURE_COLS)

    # --- 9. Sanity checks ---
    print()
    if acc > 0.98:
        print("[WARNING] Accuracy > 98% — possible overfitting to rule-based labels.")
        print("          Add more diverse video recordings or manually review labels.")
    elif acc < 0.70:
        print("[WARNING] Accuracy < 70% — model may be underfitting.")
        print("          Try increasing n_estimators or max_depth.")
    else:
        print(f"[OK] Accuracy {acc:.1%} is in the realistic range (70–98%).")

    # --- 10. Save model ---
    joblib.dump(model, model_path)
    print(f"\n[DONE] Model saved : {model_path}")
    print(f"[DONE] Features    : {len(FEATURE_COLS)}")
    print(f"\nNext step: run  streamlit run app_yolo.py")


if __name__ == "__main__":
    main()