"""
routes/pose.py
==============
FastAPI router for the Pose Safety Detection module.

All pose-related endpoints live here so main.py stays thin.
Register with:    app.include_router(pose_router)

Endpoint
--------
POST /detect/pose
    Accepts an uploaded video file.
    Runs the full pose-safety pipeline (YOLOv8s-pose + ML classifier).
    Returns a structured JSON safety summary.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.pose_service import process_pose_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

pose_router = APIRouter(prefix="/detect", tags=["Pose Safety Detection"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent.parent   # …/backend/
_TEMP_INPUT  = str(_HERE / "temp" / "upload_pose.mp4")
_TEMP_OUT    = str(_HERE / "temp" / "output" / "pose_annotated.mp4")

# Ensure output directory exists
(_HERE / "temp" / "output").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Allowed file types
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".avi"}
_ALLOWED_MIME = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "application/octet-stream",
}


def _validate_video(file: UploadFile) -> None:
    """Raise HTTPException (415) if the uploaded file is not a recognised video."""
    ext  = Path(file.filename or "").suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in _ALLOWED_EXTENSIONS and mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file '{file.filename}' (type: {mime}). "
                "Please upload a video file (.mp4, .avi, .mov)."
            ),
        )


def _cleanup(path: str) -> None:
    """Silently remove a temp file."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not delete temp file '%s': %s", path, exc)


# ---------------------------------------------------------------------------
# POST /detect/pose
# ---------------------------------------------------------------------------

@pose_router.post(
    "/pose",
    summary="Analyse worker pose safety in a video",
    response_description="Ergonomic safety summary with frame-level violations",
)
async def detect_pose(
    video: UploadFile = File(
        ...,
        description="MP4/AVI video to analyse for ergonomic / pose safety.",
    ),
):
    """
    ## Worker Pose Safety Analysis

    Upload a workplace video. The pipeline:

    1. **YOLOv8s-pose** detects human keypoints frame-by-frame.
    2. **Feature engineering** computes joint angles (back, knee, neck, elbow),
       normalised coordinates, and temporal velocity / acceleration.
    3. **ML classifier** (XGBoost / trained model) classifies each frame as
       `SAFE`, `MODERATE`, or `UNSAFE`.
    4. **5-frame majority vote** smooths the predictions temporally.
    5. An annotated video is saved to `temp/output/pose_annotated.mp4`.

    ### Safety Thresholds
    | Joint | Unsafe threshold        |
    |-------|-------------------------|
    | Back  | angle > 40°             |
    | Knee  | angle < 100°            |
    | Neck  | angle < 130°            |

    ### Response
    ```json
    {
        "total_frames":  120,
        "safe_frames":    90,
        "unsafe_frames":  30,
        "safety_score":   75.0,
        "violations": [
            {"frame": 10, "issue": "Excessive back bending"},
            {"frame": 25, "issue": "Deep knee bend — use support"}
        ]
    }
    ```
    """
    # --- Step 1: Validate the upload ---
    _validate_video(video)

    # --- Step 2: Read & persist the upload to a temp file ---
    try:
        data = await video.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        with open(_TEMP_INPUT, "wb") as f:
            f.write(data)
        logger.info(
            "Received pose upload: '%s'  (%.2f MB)",
            video.filename, len(data) / 1_048_576,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # --- Step 3: Run the pose safety pipeline ---
    try:
        result = process_pose_video(
            video_path        = _TEMP_INPUT,
            output_video_path = _TEMP_OUT,
        )
    except FileNotFoundError as exc:
        # model.pkl or video file missing
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        # OpenCV could not open the video
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # No person detected in any frame
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Pose detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during pose analysis: {exc}",
        )
    finally:
        # Always clean up the temp input regardless of success / failure
        _cleanup(_TEMP_INPUT)

    return JSONResponse(content=result)
