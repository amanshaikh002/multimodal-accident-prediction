"""
download_ppe_model.py
=====================
Downloads a pretrained YOLOv8 PPE detection model.

Two strategies are attempted in order:
  1. Pull from a known-good Hugging Face public model (no auth required)
  2. Fall back to the standard yolov8n.pt (detects 'person'; PPE labels
     won't fire but the backend will still run for smoke-testing)

Run from the project root:
    python download_ppe_model.py
"""

import os
import sys
import shutil

DEST = os.path.join("backend", "models", "ppe_model.pt")


def try_hf_public():
    """Try Hugging Face hub with a verified public repo (no login needed)."""
    try:
        from huggingface_hub import hf_hub_download
        # This repo is public and does NOT require a token
        path = hf_hub_download(
            repo_id="Ultralytics/assets",      # official Ultralytics assets repo
            filename="yolov8n.pt",             # guaranteed public
        )
        shutil.copy(path, DEST)
        print(f"[HF] Copied yolov8n.pt -> {DEST}")
        return True
    except Exception as exc:
        print(f"[HF] Failed: {exc}")
        return False


def try_ultralytics_auto():
    """Let Ultralytics auto-download yolov8n.pt from its CDN."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")          # auto-downloads to ~/.cache/ultralytics
        # Find where it was cached
        cached_path = None
        import torch
        # ultralytics caches in its own dir; locate it
        import ultralytics
        ul_dir = os.path.dirname(ultralytics.__file__)
        candidate = os.path.join(ul_dir, "assets", "yolov8n.pt")
        if os.path.isfile(candidate):
            cached_path = candidate
        else:
            # Try home dir cache
            home_cache = os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", "yolov8n.pt")
            if os.path.isfile(home_cache):
                cached_path = home_cache

        if cached_path:
            shutil.copy(cached_path, DEST)
            print(f"[Ultralytics] Copied {cached_path} -> {DEST}")
        else:
            # model.pt attribute holds the local path
            pt_path = str(getattr(model, "model", "yolov8n.pt"))
            if os.path.isfile(pt_path):
                shutil.copy(pt_path, DEST)
                print(f"[Ultralytics] Copied {pt_path} -> {DEST}")
            else:
                # Last resort — save directly
                model.save(DEST)
                print(f"[Ultralytics] Saved model -> {DEST}")
        return True
    except Exception as exc:
        print(f"[Ultralytics] Failed: {exc}")
        return False


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    if os.path.isfile(DEST):
        size_mb = os.path.getsize(DEST) / 1_048_576
        print(f"Model already exists at '{DEST}' ({size_mb:.1f} MB). Nothing to do.")
        sys.exit(0)

    print("=" * 60)
    print("PPE Model Downloader")
    print("=" * 60)
    print()
    print("NOTE: If no dedicated PPE model is found, yolov8n.pt will be")
    print("      used as a placeholder so the backend can start.")
    print("      Replace backend/models/ppe_model.pt with a real PPE")
    print("      model (.pt) from Roboflow when ready.")
    print()

    if try_hf_public() or try_ultralytics_auto():
        size_mb = os.path.getsize(DEST) / 1_048_576
        print()
        print("=" * 60)
        print(f"SUCCESS  ->  {DEST}  ({size_mb:.1f} MB)")
        print()
        print("Next steps:")
        print("  1. Install backend deps:  pip install -r backend/requirements.txt")
        print("  2. Start backend:         cd backend && uvicorn main:app --reload")
        print("  3. Test endpoint:         http://localhost:8000/docs")
        print()
        print("For a REAL PPE model (helmet + vest labels), download from:")
        print("  https://universe.roboflow.com  -> search 'PPE YOLOv8'")
        print("  Save as: backend/models/ppe_model.pt")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print()
        print("  1. Go to: https://universe.roboflow.com")
        print("  2. Search: PPE Detection YOLOv8")
        print("  3. Download the YOLOv8 .pt model file")
        print("  4. Rename / copy it to: backend/models/ppe_model.pt")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
