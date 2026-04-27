"""
routes/detect.py
================
Unified AI detection router for the Industrial Safety System.

Single entry-point:
    POST /detect?mode=<module>

Supported modes
---------------
  ppe      — PPE compliance detection  (routes to ppe_service)
  pose     — Pose safety detection     (routes to pose_service)
  fire     — Fire hazard detection     (routes to fire_service)  🔥 NEW
  combined — PPE + Pose merged pass
  all      — PPE + Pose + Fire         (full platform analysis)  🔥 NEW
  sound    — Anomaly sound detection   (placeholder — Phase 3)

This router deprecates the individual /detect/ppe, /detect/pose, and
/detect/fire endpoints. Those routes still exist for backward compat.

Usage (frontend)
----------------
    fetch('/detect?mode=ppe',  { method: 'POST', body: formData })
    fetch('/detect?mode=pose', { method: 'POST', body: formData })
    fetch('/detect?mode=fire', { method: 'POST', body: formData })
    fetch('/detect?mode=all',  { method: 'POST', body: formData })
"""

import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from services.ppe_service      import run_ppe_detection
from services.pose_service     import process_pose_video
from services.fire_service     import process_fire_video
from services.combined_service import process_combined_video, get_final_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

detect_router = APIRouter(tags=["Unified Detection"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE              = Path(__file__).resolve().parent.parent   # …/backend/
_TEMP_INPUT        = str(_HERE / "temp" / "input_video.mp4")
_PPE_OUT           = str(_HERE / "temp" / "output" / "ppe_annotated.mp4")
_POSE_OUT          = str(_HERE / "temp" / "output" / "pose_annotated.mp4")
_FIRE_OUT          = str(_HERE / "temp" / "output" / "fire_annotated.mp4")
_COMBINED_OUT      = str(_HERE / "temp" / "output" / "combined_annotated.mp4")
_ALL_OUT           = str(_HERE / "temp" / "output" / "all_annotated.mp4")

(_HERE / "temp" / "output").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Supported modes
# ---------------------------------------------------------------------------

_SUPPORTED_MODES = {"ppe", "pose", "sound", "combined", "fire", "all"}

_ALLOWED_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".avi"}
_ALLOWED_MIME = {
    "video/mp4", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "application/octet-stream",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_mode(mode: str) -> None:
    """Raise HTTP 400 if the mode is not recognised."""
    if mode not in _SUPPORTED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid mode '{mode}'. "
                f"Supported modes: {sorted(_SUPPORTED_MODES)}."
            ),
        )


def _validate_video(file: UploadFile) -> None:
    """Raise HTTP 415 if the uploaded file is not a recognised video."""
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


async def _save_upload(video: UploadFile, dest: str) -> None:
    """Read the uploaded file and persist it to dest."""
    data = await video.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    with open(dest, "wb") as f:
        f.write(data)
    logger.info(
        "Saved upload: '%s'  (%.2f MB) → %s",
        video.filename, len(data) / 1_048_576, dest,
    )


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

@detect_router.post(
    "/detect",
    summary="Unified AI safety detection",
    response_description="JSON result from the selected detection module",
)
async def detect(
    mode: str = Query(
        ...,
        description=(
            "Detection module to use. "
            "Accepted values: **ppe**, **pose**, **sound**."
        ),
        example="ppe",
    ),
    video: UploadFile = File(
        ...,
        description="Video file to analyse (.mp4, .avi, .mov).",
    ),
):
    """
    ## Unified AI Safety Detection

    A single endpoint that routes to the correct AI module based on the
    `mode` query parameter. This is the primary interface for the frontend
    dropdown selector.

    | `mode`  | Module                         | Output key        |
    |---------|--------------------------------|-------------------|
    | `ppe`   | PPE compliance detection       | `compliance_score`|
    | `pose`  | Ergonomic pose safety analysis | `safety_score`    |
    | `sound` | Anomaly sound detection        | *(coming soon)*   |

    ### PPE response
    ```json
    {
        "total_frames": 120,
        "safe_frames": 90,
        "unsafe_frames": 30,
        "compliance_score": 75.0,
        "violations": [{"frame": 10, "missing": ["helmet"]}]
    }
    ```

    ### Pose response
    ```json
    {
        "total_frames": 120,
        "safe_frames": 90,
        "unsafe_frames": 30,
        "safety_score": 75.0,
        "violations": [{"frame": 10, "issue": "Excessive back bending"}]
    }
    ```

    ### Sound response *(placeholder)*
    ```json
    {"status": "coming_soon", "message": "Sound detection — Phase 3"}
    ```
    """
    # --- Validate mode and file type ---
    mode = mode.strip().lower()
    _validate_mode(mode)
    _validate_video(video)

    # --- Save upload to shared temp path ---
    try:
        await _save_upload(video, _TEMP_INPUT)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # --- Dispatch to the selected module ---
    try:
        if mode == "ppe":
            result = run_ppe_detection(
                video_path        = _TEMP_INPUT,
                output_video_path = _PPE_OUT,
            )

        elif mode == "pose":
            result = process_pose_video(
                video_path        = _TEMP_INPUT,
                output_video_path = _POSE_OUT,
            )

        elif mode == "fire":
            result = process_fire_video(
                video_path        = _TEMP_INPUT,
                output_video_path = _FIRE_OUT,
            )

        elif mode == "combined":
            result = process_combined_video(
                video_path        = _TEMP_INPUT,
                ppe_output_path   = _COMBINED_OUT,
                pose_output_path  = _COMBINED_OUT,
            )

        elif mode == "all":
            # Full platform: PPE + Pose + Fire — run sequentially, merge results
            logger.info("[ALL] Running full platform analysis (PPE + Pose + Fire)...")
            ppe_result = run_ppe_detection(
                video_path        = _TEMP_INPUT,
                output_video_path = None,          # metrics only
            )
            pose_result = process_pose_video(
                video_path        = _TEMP_INPUT,
                output_video_path = None,          # metrics only
            )
            fire_result = process_fire_video(
                video_path        = _TEMP_INPUT,
                output_video_path = _FIRE_OUT,     # annotated fire video as primary output
            )
            # Derive module statuses
            ppe_status      = "SAFE" if (ppe_result.get("compliance_score", 0) >= 70) else "UNSAFE"
            pose_st_raw     = pose_result.get("safety_score", 0)
            pose_status     = "SAFE" if pose_st_raw >= 80 else ("MODERATE" if pose_st_raw >= 50 else "UNSAFE")
            fire_status     = fire_result.get("status", "SAFE")
            accident_status = pose_result.get("accident_status", "SAFE")
            accident_events = pose_result.get("accident_events", [])

            final_status = get_final_status(
                ppe_status, pose_status, fire_status, accident_status,
            )

            result = {
                "mode":             "all",
                "final_status":     final_status,
                "ppe_status":       ppe_status,
                "pose_status":      pose_status,
                "fire_status":      fire_status,
                "accident_status":  accident_status,
                "accident_events":  accident_events,
                "ppe_score":        ppe_result.get("compliance_score", 0.0),
                "pose_score":       pose_result.get("safety_score",    0.0),
                "fire_ratio":       fire_result.get("fire_ratio",      0.0),
                "fire_frames":      fire_result.get("fire_frames",     0),
                "total_frames":     max(
                    ppe_result.get("total_frames",  0),
                    pose_result.get("total_frames", 0),
                    fire_result.get("total_frames", 0),
                ),
                "violations":       ppe_result.get("violations", []) + pose_result.get("violations", []),
                "output_video":     "fire_annotated.mp4",
            }

        elif mode == "sound":
            # Phase 3 placeholder — returns a structured stub so the
            # frontend can handle it gracefully without a 4xx error.
            result = {
                "status":  "coming_soon",
                "mode":    "sound",
                "message": (
                    "Anomaly sound detection is not yet implemented. "
                    "It will be available in Phase 3."
                ),
            }

    except FileNotFoundError as exc:
        # Model weights or video file missing
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
        # E.g. no person detected in pose analysis
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("[%s] detection failed: %s", mode.upper(), exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during {mode} detection: {exc}",
        )
    finally:
        # Always clean up the shared temp input
        _cleanup(_TEMP_INPUT)

    # Attach the output video filename so the frontend can load it
    if mode == "ppe":
        result["video_output"] = "ppe_annotated.mp4"
    elif mode == "pose":
        result["video_output"] = "pose_annotated.mp4"
    elif mode == "fire":
        result["video_output"] = "fire_annotated.mp4"
    elif mode == "combined":
        result["video_output"] = "combined_annotated.mp4"
    elif mode == "all":
        result["video_output"] = result.get("output_video", "fire_annotated.mp4")

    logger.info("[%s] detection complete — returning result.", mode.upper())
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# GET /detect/modes  — list available modes (useful for frontend dropdowns)
# ---------------------------------------------------------------------------

@detect_router.get(
    "/detect/modes",
    summary="List available detection modes",
    tags=["Unified Detection"],
)
async def list_modes():
    """
    Returns the list of supported detection modes and their current status.
    Useful for dynamically populating frontend dropdowns.
    """
    return {
        "modes": [
            {"id": "ppe",      "label": "PPE Compliance Detection",        "status": "active"},
            {"id": "pose",     "label": "Pose Safety Detection",            "status": "active"},
            {"id": "fire",     "label": "Fire Hazard Detection",            "status": "active"},
            {"id": "combined", "label": "PPE + Pose Detection",             "status": "active"},
            {"id": "all",      "label": "Full Platform (PPE + Pose + Fire)", "status": "active"},
            {"id": "sound",    "label": "Anomaly Sound Detection",          "status": "coming_soon"},
        ]
    }
