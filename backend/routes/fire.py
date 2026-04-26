"""
routes/fire.py
==============
FastAPI router for the Fire Detection module (legacy dedicated route).

All fire-related endpoints live here so main.py stays thin.
Register with:    app.include_router(fire_router)

Endpoint
--------
POST /detect/fire
    Accepts an uploaded video file.
    Runs the fire detection pipeline (YOLOv8 fire_best.pt).
    Returns a structured JSON fire-hazard summary.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.fire_service import process_fire_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

fire_router = APIRouter(prefix="/detect", tags=["Fire Detection"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent.parent   # …/backend/
_TEMP_INPUT  = str(_HERE / "temp" / "upload_fire.mp4")
_TEMP_OUT    = str(_HERE / "temp" / "output" / "fire_annotated.mp4")

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
# POST /detect/fire
# ---------------------------------------------------------------------------

@fire_router.post(
    "/fire",
    summary="Detect fire hazards in a video",
    response_description="Fire hazard summary with frame-level detection statistics",
)
async def detect_fire(
    video: UploadFile = File(
        ...,
        description="MP4/AVI video to analyse for fire / flame hazards.",
    ),
):
    """
    ## Fire Hazard Detection

    Upload a workplace video. The pipeline:

    1. **YOLOv8 (fire_best.pt)** detects fire / flames in every Nth frame.
    2. Annotated output video is written with bounding boxes and alert banners.
    3. Returns a hazard summary including the fire ratio.

    ### Safety Logic
    - A frame is flagged if **fire / flame** is detected with confidence ≥ 0.40.
    - Status is **UNSAFE** when fire is present in > 5% of processed frames.

    ### Response
    ```json
    {
        "module":       "fire",
        "status":       "UNSAFE",
        "fire_ratio":   0.42,
        "total_frames": 120,
        "fire_frames":  50,
        "output_video": "fire_annotated.mp4"
    }
    ```
    """
    # --- validate ---
    _validate_video(video)

    # --- read & save upload ---
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
            "Received fire upload: '%s'  (%.2f MB)",
            video.filename, len(data) / 1_048_576,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save fire upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # --- run detection ---
    try:
        result = process_fire_video(
            video_path        = _TEMP_INPUT,
            output_video_path = _TEMP_OUT,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Fire detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during fire detection: {exc}",
        )
    finally:
        _cleanup(_TEMP_INPUT)

    return JSONResponse(content=result)
