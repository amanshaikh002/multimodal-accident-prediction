"""
routes/ppe.py
=============
FastAPI router for the PPE Detection module.

All PPE-related endpoints live here so main.py stays thin.
Register with:    app.include_router(ppe_router)
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.ppe_service import run_ppe_detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ppe_router = APIRouter(prefix="/detect", tags=["PPE Detection"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent.parent   # backend/
_TEMP_INPUT = str(_HERE / "temp" / "upload_ppe.mp4")
_TEMP_OUT   = str(_HERE / "temp" / "output" / "ppe_annotated.mp4")

(_HERE / "temp" / "output").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Allowed file types
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".avi"}
_ALLOWED_MIME       = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "application/octet-stream",
}


def _validate_video(file: UploadFile) -> None:
    """Raise HTTPException if the uploaded file is not a video."""
    ext  = Path(file.filename or "").suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in _ALLOWED_EXTENSIONS and mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file '{file.filename}' (type: {mime}). "
                f"Please upload a video file (.mp4, .avi, .mov)."
            ),
        )


def _cleanup(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not delete temp file '%s': %s", path, exc)


# ---------------------------------------------------------------------------
# POST /detect/ppe
# ---------------------------------------------------------------------------

@ppe_router.post(
    "/ppe",
    summary="Detect PPE compliance in a video",
    response_description="Compliance summary with frame-level violations",
)
async def detect_ppe(
    video: UploadFile = File(
        ...,
        description="MP4/AVI video to analyse for PPE compliance.",
    ),
):
    """
    ## PPE Compliance Analysis

    Upload a video file. YOLOv8 detects **human, helmet, vest, gloves, boots**
    in every Nth frame and returns a compliance summary.

    ### Safety Logic
    - A frame is **SAFE** when a person is visible AND both `helmet` + `vest` are detected.
    - A frame is **UNSAFE** when a person is present but required PPE is missing.

    ### Response
    ```json
    {
        "total_frames":     120,
        "safe_frames":       80,
        "unsafe_frames":     40,
        "compliance_score":  66.6,
        "violations": [
            {"frame": 12, "missing": ["helmet"]}
        ]
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
            "Received PPE upload: '%s'  (%.2f MB)",
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

    # --- run detection ---
    try:
        result = run_ppe_detection(
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
        logger.exception("PPE detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during PPE detection: {exc}",
        )
    finally:
        _cleanup(_TEMP_INPUT)

    return JSONResponse(content=result)
