"""
main.py
=======
FastAPI application entry-point for the Industrial Safety System.

Primary endpoint (v3)
---------------------
  POST /detect?mode=ppe   — PPE compliance detection
  POST /detect?mode=pose  — Pose safety detection
  POST /detect?mode=sound — Anomaly sound detection (placeholder)

  GET  /detect/modes      — List available modes (for frontend dropdowns)

Legacy endpoints (deprecated — kept for backward compatibility)
---------------------------------------------------------------
  POST /detect/ppe        — PPE compliance detection
  POST /detect/pose       — Pose safety detection

Run
---
    # from inside backend/
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # from project root
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from pathlib import Path

# Ensure backend/ is on sys.path for absolute imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- Routers ----------
from routes.detect import detect_router          # unified  (PRIMARY)
from routes.ppe    import ppe_router             # legacy   (deprecated)
from routes.pose   import pose_router            # legacy   (deprecated)
from routes.fire   import fire_router            # legacy   (deprecated)
from routes.sound  import sound_router           # legacy   (deprecated)

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
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Industrial Safety System — API",
    description=(
        "Modular AI-powered safety detection backend.\n\n"
        "## Primary Endpoint (recommended)\n"
        "- 🎯 **Unified Detection** — `POST /detect?mode=<ppe|pose|fire|sound|combined|all>`\n"
        "- 📋 **List Modes** — `GET /detect/modes`\n\n"
        "## Legacy Endpoints *(deprecated)*\n"
        "- 🦺 PPE Detection — `POST /detect/ppe`\n"
        "- 🧍 Pose Safety Detection — `POST /detect/pose`\n"
        "- 🔥 Fire Hazard Detection — `POST /detect/fire`\n"
        "- 🔊 Anomaly Sound Detection — `POST /detect/sound`"
    ),
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React / Vite dev servers
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # CRA
        "http://localhost:5173",   # Vite (default)
        "http://localhost:5174",   # Vite (fallback when 5173 is busy)
        "http://localhost:5175",   # Vite (fallback)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

# Primary unified router (v3)
app.include_router(detect_router)

# Legacy module routers — kept for backward compatibility only.
# These will be removed in a future version.
app.include_router(ppe_router)
app.include_router(pose_router)
app.include_router(fire_router)
app.include_router(sound_router)

# ---------------------------------------------------------------------------
# Static files — serve annotated output videos to the frontend
# ---------------------------------------------------------------------------

_OUTPUT_DIR = _HERE / "temp" / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(_OUTPUT_DIR)), name="output")

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="Root health check")
async def root():
    return {
        "status":  "ok",
        "message": "Industrial Safety System API v3 is running.",
        "docs":    "/docs",
        "primary_endpoint": "POST /detect?mode=<ppe|pose|sound>",
    }


@app.get("/health", tags=["Health"], summary="Module status")
async def health():
    return {
        "status":  "healthy",
        "version": "3.2.0",
        "modules": {
            "ppe_detection":   "active",
            "pose_detection":  "active",
            "fire_detection":  "active",
            "sound_detection": "active",
            "combined_mode":   "active",
            "all_mode":        "active",
        },
        "endpoints": {
            "unified":      "POST /detect?mode=<module>   <- recommended",
            "list_modes":   "GET  /detect/modes",
            "legacy_ppe":   "POST /detect/ppe            <- deprecated",
            "legacy_pose":  "POST /detect/pose           <- deprecated",
            "legacy_fire":  "POST /detect/fire           <- deprecated",
            "legacy_sound": "POST /detect/sound          <- deprecated",
        },
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
