"""
main.py
=======
FastAPI application entry-point for the Industrial Safety System backend.

Modules exposed:
  POST /detect/ppe  — PPE compliance detection from video upload

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

CORS is pre-configured to allow the React dev server (localhost:3000 / 5173).
"""

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the backend directory is on sys.path so that absolute imports work
# when running uvicorn from inside `backend/` OR from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from services.ppe_service import run_ppe_detection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
TEMP_DIR   = _HERE / "temp"
OUTPUT_DIR = _HERE / "temp" / "output"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_VIDEO_PATH  = str(TEMP_DIR / "temp_ppe.mp4")
OUTPUT_VIDEO_PATH = str(OUTPUT_DIR / "ppe_annotated.mp4")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Industrial Safety System — Backend API",
    description=(
        "Modular safety detection backend supporting:\n"
        "- **PPE Detection** (Phase 1)\n"
        "- Pose Detection (Phase 2, coming soon)\n"
        "- Anomaly Sound Detection (Phase 3, coming soon)"
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the React frontend (localhost:3000 and Vite default :5173)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Server health check."""
    return {"status": "ok", "message": "Industrial Safety System API is running."}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "modules": {
            "ppe_detection": "active",
            "pose_detection": "coming_soon",
            "anomaly_sound": "coming_soon",
        },
    }


# ---------------------------------------------------------------------------
# PPE Detection endpoint
# ---------------------------------------------------------------------------

_ALLOWED_MIME_TYPES = {"video/mp4", "video/mpeg", "video/quicktime"}
_ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mov"}
_MAX_FILE_SIZE_MB   = 500  # soft limit (informational only; actual limit TBD by server)


@app.post(
    "/detect/ppe",
    tags=["PPE Detection"],
    summary="Detect PPE compliance in an uploaded video",
    response_description="Frame-wise detections + video-level compliance summary",
)
async def detect_ppe(
    video: UploadFile = File(
        ...,
        description="MP4 video file to analyse for PPE compliance (helmet + vest).",
    ),
):
    """
    **PPE Detection Endpoint**

    Accepts an MP4 video upload, runs YOLOv8 inference on every 2nd frame,
    and returns:

    - Per-frame detections (label, confidence, bounding box)
    - Per-frame PPE status (helmet ✓/✗, vest ✓/✗)
    - Video-level compliance metrics (compliance ratio, final status)

    **Status values**:
    - `PPE Compliant`  — both helmet and vest detected in ≥ 50 % of frames
    - `Helmet Missing` — helmet detected in < 50 % of frames
    - `Vest Missing`   — vest detected in < 50 % of frames

    **Response structure**:
    ```json
    {
        "mode": "ppe",
        "summary": {
            "total_frames": 120,
            "helmet_frames": 110,
            "vest_frames": 95,
            "helmet_compliance": 0.9167,
            "vest_compliance": 0.7917,
            "status": "PPE Compliant"
        },
        "frames": [
            {
                "frame_id": 0,
                "helmet": true,
                "vest": true,
                "detections": [
                    {"label": "helmet", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
                ]
            }
        ]
    }
    ```
    """
    # ------------------------------------------------------------------
    # 1. Validate file type
    # ------------------------------------------------------------------
    filename  = video.filename or ""
    extension = Path(filename).suffix.lower()
    content_type = (video.content_type or "").lower()

    if extension not in _ALLOWED_EXTENSIONS and content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{extension}' / content-type '{content_type}'. "
                f"Please upload an MP4 video file."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Save uploaded file to temp directory
    # ------------------------------------------------------------------
    logger.info("Received PPE detection request — file: '%s'", filename)
    try:
        content = await video.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        with open(TEMP_VIDEO_PATH, "wb") as f:
            f.write(content)

        logger.info("Saved upload to: %s (%.2f MB)", TEMP_VIDEO_PATH, len(content) / 1_048_576)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # ------------------------------------------------------------------
    # 3. Run PPE detection service
    # ------------------------------------------------------------------
    try:
        result = run_ppe_detection(
            video_path=TEMP_VIDEO_PATH,
            output_video_path=OUTPUT_VIDEO_PATH,
        )
    except FileNotFoundError as exc:
        # Model file missing
        logger.error("Model not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        # Bad video (corrupt / empty)
        logger.error("Video processing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during PPE detection: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during PPE detection: {exc}",
        )
    finally:
        # ------------------------------------------------------------------
        # 4. Cleanup temp input file
        # ------------------------------------------------------------------
        _cleanup(TEMP_VIDEO_PATH)

    logger.info(
        "PPE detection finished — status: '%s', frames: %d",
        result["summary"]["status"],
        result["summary"]["total_frames"],
    )

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _cleanup(path: str) -> None:
    """Remove a temporary file without raising exceptions."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Cleaned up temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not clean up '%s': %s", path, exc)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
