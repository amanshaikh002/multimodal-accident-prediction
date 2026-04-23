"""
train_model.py — RandomForest Posture Classifier Training (Rebuilt)
====================================================================
Worker Pose Safety Monitoring System — Production Pipeline

Trains a RandomForest (primary) or XGBoost (if installed) classifier on the
feature vector produced by the YOLOv8 pipeline.

Pipeline:
  1. Load labeled_dataset.csv (balanced labels from auto_label.py)
  2. Validate & clean features
  3. Video-based train/test split (no data leakage between clips)
  4. SMOTE oversampling on the TRAIN set only (handles class imbalance)
  5. Train RandomForest with class_weight="balanced"
  6. Evaluate: accuracy, confusion matrix, classification report
  7. Print per-class prediction distribution sanity check
  8. Print feature importances
  9. Save model.pkl

Feature vector (matches utils.FEATURE_COLS — 9 stable features):
  back_angle, knee_angle, neck_angle,
  norm_shoulder_x, norm_shoulder_y,
  norm_hip_x, norm_hip_y,
  norm_knee_x, norm_knee_y

NOTE: velocity/acceleration features deliberately removed — they cause noise.

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

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_FRACTION: float = 0.20
RANDOM_STATE:  int   = 42
LABEL_MAP:  dict = {"safe": 0, "moderate": 1, "unsafe": 2}
LABEL_NAMES: dict = {0: "safe", 1: "moderate", 2: "unsafe"}
ANGLE_COLS_COUNT: int = 3   # first 3 features are angles

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def video_based_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by VIDEO SOURCE to prevent data leakage.
    Frames from the same video clip are highly correlated (consecutive frames).
    """
    video_names = df["video_name"].unique()
    n_test = max(1, int(len(video_names) * test_fraction))

    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(video_names)

    test_videos  = set(shuffled[:n_test])
    train_videos = set(shuffled[n_test:])

    train_df = df[df["video_name"].isin(train_videos)].copy()
    test_df  = df[df["video_name"].isin(test_videos)].copy()

    print(f"\n--- Video-Based Split ---")
    print(f"  Total videos : {len(video_names)}")
    print(f"  Train videos : {len(train_videos)}")
    print(f"  Test  videos : {len(test_videos)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test  samples: {len(test_df)}")

    return train_df, test_df


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the training set to address class imbalance.
    Falls back to manual upsampling if imbalanced-learn is not installed.
    """
    min_count = _min_class_count(y)

    try:
        from imblearn.over_sampling import SMOTE
        k = min(5, min_count - 1)
        if k < 1:
            print(f"[WARN] SMOTE needs k_neighbors >= 1 (min class has {min_count} samples). Using manual upsampling.")
            return _manual_upsample(X, y)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"[INFO] SMOTE applied. New training size: {len(X_res)}")
        return X_res, y_res
    except ImportError:
        print("[WARN] imbalanced-learn not installed. Using manual upsampling.")
        print("       Install:  pip install imbalanced-learn")
        return _manual_upsample(X, y)
    except Exception as e:
        print(f"[WARN] SMOTE failed ({e}). Using manual upsampling.")
        return _manual_upsample(X, y)


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _manual_upsample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bring all classes to the majority count by upsampling with replacement."""
    from sklearn.utils import resample
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())
    X_parts, y_parts = [], []
    for cls in classes:
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        if len(X_cls) < max_count:
            X_cls, y_cls = resample(
                X_cls, y_cls,
                n_samples=max_count,
                random_state=RANDOM_STATE,
            )
        X_parts.append(X_cls)
        y_parts.append(y_cls)
    return np.vstack(X_parts), np.concatenate(y_parts)


def build_model():
    """
    Primary: RandomForest with class_weight='balanced' (reliable, interpretable).
    Fallback: XGBoost with scale_pos_weight handling.
    """
    # ── Option A: XGBoost (if installed) ───────────────────────────────────
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
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

    # ── Option B: RandomForest (always available, recommended) ──────────────
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",          # ← handles imbalance automatically
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("[INFO] Using RandomForest classifier (n_estimators=100, max_depth=10, class_weight=balanced)")
    return model


def predict_with_threshold(
    classifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability-based prediction with custom decision thresholds:
      • prob_unsafe > 0.70  → UNSAFE  (2)
      • prob_safe   > 0.60  → SAFE    (0)
      • else                → MODERATE (1)

    Returns (preds, max_proba_per_sample)
    """
    if not hasattr(classifier, "predict_proba"):
        preds = classifier.predict(X)
        return preds, np.ones(len(preds))

    proba = classifier.predict_proba(X)  # shape (N, n_classes)
    n_classes = proba.shape[1]

    preds = []
    for p in proba:
        # Map class indices to probabilities safely
        prob_safe     = float(p[0]) if n_classes > 0 else 0.0
        prob_moderate = float(p[1]) if n_classes > 1 else 0.0
        prob_unsafe   = float(p[2]) if n_classes > 2 else 0.0

        if prob_unsafe > 0.70:
            preds.append(2)
        elif prob_safe > 0.60:
            preds.append(0)
        else:
            preds.append(1)

    preds_arr  = np.array(preds)
    max_proba  = proba.max(axis=1)
    return preds_arr, max_proba


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
    base_dir   = Path(__file__).resolve().parent
    input_csv  = base_dir / "labeled_dataset.csv"
    model_path = base_dir / "model.pkl"

    # --- 1. Load data ---
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {input_csv}\n"
            "Run auto_label.py first."
        )

    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} rows from {input_csv.name}")
    print(f"[INFO] FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # --- 2. Validate columns ---
    required = set(FEATURE_COLS) | {"label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in labeled_dataset.csv: {sorted(missing)}\n"
            "Re-run auto_label.py to regenerate labeled_dataset.csv."
        )

    # Convert to numeric, drop NaN
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=FEATURE_COLS + ["label"])
    df = df[df["label"].isin(LABEL_MAP)].copy()

    if df.empty:
        raise ValueError("No valid rows after cleaning. Check your labeled_dataset.csv.")

    df["label_encoded"] = df["label"].map(LABEL_MAP)

    print(f"[INFO] Total clean samples: {len(df)}")

    # --- Class distribution ---
    print("\n--- Class Distribution (raw) ---")
    for cls_id in [0, 1, 2]:
        count = int((df["label_encoded"] == cls_id).sum())
        pct   = 100.0 * count / len(df)
        print(f"  {LABEL_NAMES[cls_id]:8s}: {count:5d}  ({pct:.1f}%)")

    # --- Angle statistics ---
    print("\n--- Angle Statistics ---")
    for col in ["back_angle", "knee_angle", "neck_angle"]:
        if col in df.columns:
            print(f"  {col:12s}: min={df[col].min():.1f}°  mean={df[col].mean():.1f}°  max={df[col].max():.1f}°")

    # --- 3. Train/test split ---
    if "video_name" in df.columns and df["video_name"].nunique() > 1:
        train_df, test_df = video_based_split(df)

        # Fallback to random split if test set too small / mono-class
        if len(test_df) < 15 or test_df["label_encoded"].nunique() < 2:
            print("[WARN] Video split insufficient — falling back to stratified random split.")
            X_all  = df[FEATURE_COLS].values
            y_all  = df["label_encoded"].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_FRACTION,
                random_state=RANDOM_STATE, stratify=y_all,
            )
        else:
            X_train = train_df[FEATURE_COLS].values
            y_train = train_df["label_encoded"].values
            X_test  = test_df[FEATURE_COLS].values
            y_test  = test_df["label_encoded"].values
    else:
        X_all  = df[FEATURE_COLS].values
        y_all  = df["label_encoded"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=TEST_FRACTION,
            random_state=RANDOM_STATE, stratify=y_all,
        )

    # --- 4. SMOTE oversampling on TRAIN ONLY ---
    X_train, y_train = apply_smote(X_train, y_train)

    print(f"\n[INFO] Train samples (after SMOTE): {len(X_train)}")
    print(f"[INFO] Test  samples (clean)       : {len(X_test)}")

    print("\n--- Train distribution (after SMOTE) ---")
    for cls_id in [0, 1, 2]:
        cnt = int((y_train == cls_id).sum())
        print(f"  {LABEL_NAMES[cls_id]:8s}: {cnt:5d}")

    # --- 5. Build and train model ---
    model = build_model()
    print(f"\n[INFO] Training model...")
    model.fit(X_train, y_train)
    print(f"[INFO] Training complete.")

    # --- 6. Evaluate with threshold-based prediction ---
    y_pred, _ = predict_with_threshold(model, X_test)
    acc        = accuracy_score(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)

    present      = sorted(set(y_test) | set(y_pred))
    target_names = [LABEL_NAMES[c] for c in present]
    report       = classification_report(y_test, y_pred, labels=present, target_names=target_names)

    print(f"\n{'='*55}")
    print(f"Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"{'='*55}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join(f"{LABEL_NAMES[c]:8s}" for c in present)
    print(header)
    for i, cls in enumerate(present):
        row_label = f"{LABEL_NAMES[cls]:8s}"
        row_vals  = "  ".join(f"{cm[i][j]:8d}" for j in range(len(present)))
        print(f"  {row_label} {row_vals}")
    print("\nClassification Report:")
    print(report)

    # --- Prediction distribution sanity check ---
    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    print("\n--- Prediction Distribution on Test Set ---")
    for cls, cnt in zip(pred_classes, pred_counts):
        pct = 100.0 * cnt / len(y_pred)
        print(f"  {LABEL_NAMES[cls]:8s}: {cnt:5d}  ({pct:.1f}%)")

    if len(pred_classes) < 3:
        missing = [LABEL_NAMES[c] for c in [0, 1, 2] if c not in pred_classes]
        print(f"\n[WARNING] Model never predicts: {missing}")
        print("          Check labeling: run auto_label.py and review angle statistics.")
    elif min(pred_counts) / max(pred_counts) < 0.05:
        print("\n[WARNING] Prediction distribution is severely imbalanced.")
        print("          Review auto_label.py output and add diverse video recordings.")
    else:
        print("\n[OK] Prediction distribution looks balanced.")

    # --- 7. Feature importances ---
    print_feature_importances(model, FEATURE_COLS)

    # --- 8. Sanity checks ---
    print()
    if acc > 0.98:
        print("[WARNING] Accuracy > 98% — possible overfitting. Add more diverse videos.")
    elif acc < 0.60:
        print("[WARNING] Accuracy < 60% — model may be underfitting.")
        print("          Try adding more training videos or review labeling rules.")
    else:
        print(f"[OK] Accuracy {acc:.1%} is in the realistic range (60–98%).")

    # --- 9. Save model ---
    joblib.dump(model, model_path)
    print(f"\n[DONE] Model saved : {model_path}")
    print(f"[DONE] Features    : {len(FEATURE_COLS)}  →  {FEATURE_COLS}")
    print(f"\nNext step: run  streamlit run app_yolo.py")


if __name__ == "__main__":
    main()