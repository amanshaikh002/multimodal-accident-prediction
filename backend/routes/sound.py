"""
routes/sound.py
===============
FastAPI router for the Anomaly Sound Detection module.

Endpoint
--------
POST /detect/sound
    Accepts an uploaded video file.
    Extracts the audio track and runs the sliding-window MFCC + RandomForest
    classifier (audio_model.pkl) trained on machine sound normal/abnormal.
    Returns a structured anomaly summary with per-event timestamps.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.sound_service import process_sound_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

sound_router = APIRouter(prefix="/detect", tags=["Sound Anomaly Detection"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent.parent   # ...
_TEMP_INPUT  = str(_HERE / "temp" / "upload_sound.mp4")
_TEMP_OUT    = str(_HERE / "temp" / "output" / "sound_annotated.mp4")

(_HERE / "temp" / "output").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Allowed file types
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".avi", ".mkv", ".webm"}
_ALLOWED_MIME = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/x-matroska", "video/webm",
    "application/octet-stream",
}


def _validate_video(file: UploadFile) -> None:
    ext  = Path(file.filename or "").suffix.lower()
    mime = (file.content_type or "").lower()
    if ext not in _ALLOWED_EXTENSIONS and mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file '{file.filename}' (type: {mime}). "
                "Please upload a video file (.mp4, .avi, .mov, .mkv, .webm)."
            ),
        )


def _cleanup(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not delete temp file '%s': %s", path, exc)


# ---------------------------------------------------------------------------
# POST /detect/sound
# ---------------------------------------------------------------------------

@sound_router.post(
    "/sound",
    summary="Detect anomalous sounds in a video's audio track",
    response_description="Anomaly summary with per-event timestamps",
)
async def detect_sound(
    video: UploadFile = File(
        ...,
        description="Video file whose audio track should be analysed.",
    ),
):
    """
    ## Anomaly Sound Detection

    1. Extract the audio track from the uploaded video.
    2. Slide a 3-second window over the waveform with a 1-second hop.
    3. For each window, compute MFCC mean (13-dim) and run the trained
       RandomForest classifier (`audio_model.pkl`).
    4. A window only counts as anomalous when at least 2 consecutive raw
       predictions are positive (suppresses single-window flicker).
    5. Group confirmed anomalous windows into contiguous events.

    ### Verdict
    - **UNSAFE** if more than 5% of windows are anomalous, OR any single
      event lasts >= 2 seconds.
    - **SAFE** otherwise.

    ### Response
    ```json
    {
        "module":           "sound",
        "status":           "UNSAFE",
        "anomaly_detected": true,
        "anomaly_ratio":    0.18,
        "total_windows":    28,
        "anomaly_windows":  5,
        "duration_sec":     30.0,
        "events": [
          {"start_sec": 4.0, "end_sec": 7.0, "duration_sec": 3.0,
           "avg_confidence": 0.85, "max_confidence": 0.92,
           "max_anomaly_prob": 0.92}
        ],
        "message": "...",
        "output_video": "sound_annotated.mp4"
    }
    ```
    """
    _validate_video(video)

    # ---- read & save upload ---
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
            "Received sound upload: '%s'  (%.2f MB)",
            video.filename, len(data) / 1_048_576,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save sound upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # ---- run detection ---
    try:
        result = process_sound_video(
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
        logger.exception("Sound detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during sound detection: {exc}",
        )
    finally:
        _cleanup(_TEMP_INPUT)

    return JSONResponse(content=result)
