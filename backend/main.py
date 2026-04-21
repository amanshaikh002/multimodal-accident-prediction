"""
main.py
=======
FastAPI application entry-point for the Industrial Safety System.

Registered modules
------------------
  POST /detect/ppe   — PPE compliance detection  (Phase 1 — LIVE)
  POST /detect/pose  — Pose safety detection      (Phase 2 — coming soon)
  POST /detect/sound — Anomaly sound detection    (Phase 3 — coming soon)

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

# ---------- Routers ----------
from routes.ppe import ppe_router
# from routes.pose  import pose_router    # uncomment when ready
# from routes.sound import sound_router   # uncomment when ready

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
        "**Active modules:**\n"
        "- 🦺 **PPE Detection** — `POST /detect/ppe`\n\n"
        "**Coming soon:**\n"
        "- 🧍 Pose Safety Detection — `POST /detect/pose`\n"
        "- 🔊 Anomaly Sound Detection — `POST /detect/sound`"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev servers
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # CRA default
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(ppe_router)
# app.include_router(pose_router)
# app.include_router(sound_router)

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="Root health check")
async def root():
    return {"status": "ok", "message": "Industrial Safety System API v2 is running."}


@app.get("/health", tags=["Health"], summary="Module status")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "modules": {
            "ppe_detection":        "active",
            "pose_detection":       "coming_soon",
            "anomaly_sound":        "coming_soon",
        },
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
