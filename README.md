# Multimodal Vision Audio Framework for Workplace Accident Prediction

An AI-powered industrial safety system that monitors workplaces in real-time, enforces PPE compliance, watches for unsafe postures, detects fires and worker accidents, and listens for anomalous machine sounds. The framework couples computer vision with audio analysis through a modular pipeline, separating a Python (FastAPI) backend from a unified React dashboard.

---

## Architecture Overview

The system is built as a **modular service architecture** so each safety signal evolves independently and can be combined into a unified verdict.

* **Frontend:** React 19 + Vite dashboard, a separate analysis panel per module
* **Backend:** FastAPI + Uvicorn, one service module per detection signal
* **Computer Vision:** Ultralytics YOLOv8 (PPE custom weights, YOLOv8s-pose, fire-detection HuggingFace weights, YOLO-World for open-vocab hazards)
* **ML / Ergonomics:** Scikit-Learn RandomForest (pose classifier) and a separate RandomForest for audio anomaly classification
* **Audio:** librosa + bundled imageio-ffmpeg for MFCC feature extraction
* **Hardware:** automatic CUDA fallback to CPU; OpenCV for video I/O; ffmpeg bundled via `imageio-ffmpeg` so no system PATH setup is needed

---

## Implemented Modules

The system now has **five detection modules** plus a **Full Platform** mode that bundles three of them into a single unified pass.

### 1. PPE Compliance Detection ([backend/services/ppe_service.py](backend/services/ppe_service.py))

Person-centric PPE checks with occlusion awareness, sticky-state smoothing for bent / non-frontal workers, and open-vocabulary hazard overlay.

**Detector:** custom-trained YOLOv8 (`backend/models/ppe_model.pt`) detecting `human`, `helmet`, `vest`, `gloves`, `boots`. Center-point containment is used to bind PPE items to specific workers (more reliable than IoU for small items like helmets).

**Tri-state safety verdict** — replaces the previous binary safe/unsafe contract:

| Scene | Verdict |
|---|---|
| Empty, motionless, no recently-seen worker | `SAFE` |
| Worker visible, helmet + vest detected | `SAFE` |
| Worker visible, missing helmet/vest | `UNSAFE` |
| No worker detected **but** scene has motion (debris, brick fall, etc.) | `UNKNOWN` |
| Worker seen <1.5 s ago, now hidden | `UNKNOWN` (occlusion) |

**Sticky-PPE smoothing:** the trained model has poor recall on bent / non-frontal poses (back angle, occluded torso). A required item that was visible on a worker within the last ~1.5 s is treated as still worn so a single bad-detection frame does not flip the verdict to UNSAFE.

**Open-vocabulary hazard detection (YOLO-World):** runs alongside the PPE pass on every processed frame with prompts like `["falling rock", "loose brick", "debris", "fallen person", "falling object"]`. Hazards override every other state to UNSAFE. Three filters suppress false positives:

* **Per-class confidence floor** (e.g. `fallen person` requires conf ≥ 0.55, much higher than static debris).
* **Aspect-ratio gate for `fallen person`** — bbox `w/h` must be ≥ 1.2 (lying horizontal). Bending or crouching workers stay vertical and are rejected.
* **Temporal persistence (2 consecutive frames)** — kills single-frame flicker.

**Outputs:** JSON (`safe_frames`, `unsafe_frames`, `unknown_frames`, `hazard_frames`, `compliance_score`, `violations`) plus an annotated video with green / red / yellow status banners and red hazard overlays.

### 2. Worker Pose Ergonomics & Accident Detection ([backend/services/pose_service.py](backend/services/pose_service.py))

Two parallel signals on top of the same pose pass:

#### A. Ergonomic posture classifier
* **Keypoint extraction:** `yolov8s-pose.pt` → 17 COCO keypoints
* **Feature engineering:** joint angles (back / knee / neck / elbow), normalized coords (hip-centered, torso-scaled), velocities, accelerations
* **Hybrid classifier:** rule-based geometric overrides for extreme poses, RandomForest (`ml/pose/model.pkl`) for the ambiguous middle
* **Temporal smoothing:** 5-frame majority vote, 3-consecutive-unsafe-frame gate before logging a violation

#### B. Accident / fall detector ([backend/utils/accident_detector.py](backend/utils/accident_detector.py))

A separate rule-based event detector running on top of the pose keypoints. **Fundamentally different from the ergonomic classifier** — answers "did something just happen?" rather than "is the posture good?".

| Event | Pose signature | Severity |
|---|---|---|
| `FALL` | body axis flips from vertical (≥70°) to horizontal (≤30°) **and** bbox aspect ratio collapses by ≥50% within ~1.5 s | CRITICAL |
| `STRUCK` | sudden COM acceleration spike (≥3-σ over rolling 30-frame baseline) while still upright | WARN |
| `CRUSHED` | tracked at avg conf ≥ 0.60 for 15 frames, then collapses to ≤ 0.20 for 30 frames **and** bbox shrinks to ≤ 65% of baseline | CRITICAL |
| `MOTIONLESS_DOWN` | after a FALL, body stays horizontal with near-zero COM motion for ≥3 s | CRITICAL (medical) |
| `STUMBLE` | hip-Y drops fast then recovers; body axis never fully horizontalises | WARN |

**Per-person IoU tracker** ([backend/utils/pose_tracker.py](backend/utils/pose_tracker.py)) — assigns stable IDs across frames so velocity/acceleration buffers stay locked to one worker. Without this, switching the "primary person" mid-clip would corrupt the temporal features.

**Overlay renderer:** TTL-based event persistence (~3 s on screen), pulsing red border for CRITICAL, orange for WARN, per-bbox highlight on the affected worker. Survives no-pose frames so the warning doesn't flicker when YOLO momentarily loses keypoints.

### 3. Fire Hazard Detection ([backend/services/fire_service.py](backend/services/fire_service.py))

A dedicated YOLOv8 fire/smoke model auto-downloaded from HuggingFace, with HSV verification removed in favour of a properly-trained model.

**Loader priority:**
1. `FIRE_MODEL` env var → custom local `.pt` file
2. HuggingFace fire model (default: `SalahALHaismawi/yolov26-fire-detection`, ~20 MB, classes `fire`, `other`, `smoke`) — auto-downloaded once and cached
3. YOLO-World fallback prompted with fire vocabulary (only if both above fail)

**Resolution/aspect-ratio handling:**
* The original frame is passed to YOLO with `imgsz=1280` (default). Ultralytics letterboxes internally so portrait videos like 1080×1920 stay portrait and aren't squished into landscape.
* Output video keeps the source aspect ratio, capped by `FIRE_OUTPUT_MAX_SIDE=1280` so 4K input doesn't produce a multi-GB file.
* Banner / box / font sizes scale with frame height so annotations stay readable on 1080p+.

**Reliability layer:**
* Bundled **imageio_ffmpeg** binary used for any subprocess work — no system PATH dependency on Windows
* **Temporal persistence (2 consecutive frames)** kills single-frame flicker
* **Verbose per-detection logging** (`FIRE_VERBOSE=1`, default ON) — every YOLO box emits a `[FIRE] frame=N HIT/LOW_CONF/OTHER label='...' conf=X.XXX` line for transparent debugging
* **`draw_banner=False` kwarg** lets it chain cleanly under the combined-mode banner so the FIRE/NO-FIRE banner doesn't paint over the upstream UI

### 4. Anomaly Sound Detection ([backend/services/sound_service.py](backend/services/sound_service.py))

Sliding-window machine-sound classifier. Listens for abnormal mechanical sounds (grinding, hissing, banging) that may indicate equipment failure.

**Pipeline (per video):**
1. Extract mono audio @ 22 050 Hz: `librosa.load` first, `imageio_ffmpeg` subprocess fallback
2. Slide a **3-second window** with a **1-second hop** (one prediction per second of audio)
3. For each window: compute MFCC mean (13-dim) → run trained RandomForest (`backend/Sound Anomaly/audio_model.pkl`)
4. Apply **2-window persistence** — single-window false hits are discarded
5. Group confirmed anomalous windows into contiguous events with `start_sec`, `end_sec`, `duration_sec`, `avg_confidence`, `max_confidence`

**Verdict:** `UNSAFE` if the anomaly ratio exceeds 5% of windows OR any single event lasts ≥2 s, else `SAFE`.

**Graceful failure:** every audio extraction error (no audio track, codec issue, missing ffmpeg, corrupt file) returns 200 OK with a clear message — never a 4xx. The sound module is the safest to attach because a video without audio still produces a valid response.

### 5. Combined Multimodal Pipeline ([backend/services/combined_service.py](backend/services/combined_service.py))

Synchronized PPE + Pose pass: the pose model gives keypoints, the PPE model runs on the same clean frame, and a single annotated video is rendered with skeleton + per-person bounding box (color-coded by SAFE/MODERATE/UNSAFE/HIGH-RISK), accident overlays, and a global safety banner.

**Sticky-PPE smoothing** is also active here so brief detection misses don't flicker the per-person box to UNSAFE.

### 6. Full Platform Mode (the `all` route)

The `mode=all` endpoint runs **PPE + Pose + Fire** in a single bundle (sound is its own dedicated mode and is intentionally not included). The pipeline is:

1. Combined service writes `combined_annotated.mp4` (skeleton + PPE boxes + accident overlays + global banner)
2. Fire service runs **on that already-annotated output** with `draw_banner=False`, overlaying fire bounding boxes only
3. Final output: `all_annotated.mp4` containing PPE + Pose + Accident + Fire annotations in one file

---

## Unified Decision Engine

The combined verdict across modules follows this priority hierarchy:

```
CRITICAL    fire UNSAFE  OR  accident_status == CRITICAL
HIGH RISK   PPE UNSAFE  AND  Pose UNSAFE
UNSAFE      PPE UNSAFE  OR  Pose UNSAFE  OR  accident_status == WARN
            OR  sound UNSAFE
MODERATE    Pose MODERATE only
SAFE        all clear
```

The frontend `AllPanel` reads `fire_status` and `accident_status` to decide whether a CRITICAL banner should say *"FIRE HAZARD"*, *"WORKER ACCIDENT"*, or *"FIRE & WORKER ACCIDENT"* — never just defaulting to "FIRE" anymore.

---

## Frontend Dashboard ([frontend/](frontend/))

A React + Vite single-page dashboard. Module dropdown drives the pipeline; the right column adapts to whichever mode was last analysed.

| Component | What it shows |
|---|---|
| `AnalysisPanel` | PPE-only / Pose-only summary (compliance %, violations) |
| `CombinedPanel` | PPE + Pose merged metrics |
| `FirePanel` | Fire status + ratio + frame counts |
| `SoundPanel` | Sound status, anomaly ratio, **timeline bar** with red bands per anomalous event, scrollable event list with timestamps + confidences |
| `AllPanel` | Full-platform summary, dynamic CRITICAL labelling, optional **Accident Events** card |
| `ViolationsList` | Frame-by-frame violations from any module |
| `Recommendations` | Actionable text per detected issue |

**Audio alerts** (`useAudioAlerts` hook):
* **Generic pose alert** — *"Warning! Unsafe body posture detected. Review worker form."* — fires once per analysis. The previous activity-specific phrasing ("unsafe lifting posture") was removed because the model can't tell whether the worker is actually lifting.
* **PPE alerts** stay specific (helmet / vest / gloves / boots).
* **Fire / sound / accident alerts** are urgent and event-specific.

---

## Project Structure

```text
├── frontend/                          # React 19 + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                    # Top-level layout + routing per mode
│   │   ├── components/
│   │   │   ├── AnalysisPanel.jsx      # PPE / Pose summary
│   │   │   ├── CombinedPanel.jsx      # PPE + Pose merged
│   │   │   ├── FirePanel.jsx          # Fire status + stats
│   │   │   ├── SoundPanel.jsx         # Anomaly sound timeline + events
│   │   │   ├── AllPanel.jsx           # Full Platform summary (dynamic CRITICAL label)
│   │   │   ├── VideoPanel.jsx         # Annotated-video player
│   │   │   ├── ViolationsList.jsx     # Frame-by-frame violations
│   │   │   └── Recommendations.jsx    # Per-issue suggestions
│   │   ├── hooks/
│   │   │   └── useAudioAlerts.js      # SpeechSynthesis-based voice alerts
│   │   └── suggestionMap.js           # Backend reasons -> human-readable enrichments
│   └── package.json
│
├── backend/                           # Modular FastAPI backend
│   ├── main.py                        # App entry, CORS, router registration, /health
│   ├── requirements.txt
│   ├── models/                        # ppe_model.pt, yolov8s-pose.pt, model.pkl
│   ├── Sound Anomaly/
│   │   └── audio_model.pkl            # RandomForest audio classifier
│   ├── routes/
│   │   ├── detect.py                  # Unified router: POST /detect?mode=...
│   │   ├── ppe.py / pose.py / fire.py / sound.py     # Legacy dedicated routes
│   ├── services/
│   │   ├── ppe_service.py             # PPE pipeline (tri-state, sticky, hazard)
│   │   ├── pose_service.py            # Pose pipeline (ergonomic + accident detection)
│   │   ├── fire_service.py            # HF-fire-model + YOLO-World fallback
│   │   ├── sound_service.py           # Sliding-window MFCC audio anomaly
│   │   └── combined_service.py        # PPE + Pose merged + final-status engine
│   └── utils/
│       ├── ppe_utils.py               # Tri-state evaluator, motion score, hazard override
│       ├── pose_utils.py              # Joint angles, hybrid classifier, drawing helpers
│       ├── pose_tracker.py            # IoU per-person tracker (stable IDs)
│       ├── accident_detector.py       # FALL / STRUCK / CRUSHED / MOTIONLESS / STUMBLE
│       └── video_utils.py             # OpenCV I/O, ffmpeg H.264 re-encode
│
├── ml/                                # Training pipelines
│   └── pose/
│       ├── auto_label.py              # Geometric labeler for skeletal training data
│       ├── dataset_generator_yolo.py  # Feature extraction over labelled videos
│       ├── train_model.py             # RandomForest / XGBoost training, video-level split
│       └── model.pkl                  # Trained ergonomic classifier
│
└── implementation_details.md          # Detailed phase-by-phase notes
```

---

## API Endpoints

### Primary unified endpoint

```
POST /detect?mode=<ppe|pose|fire|sound|combined|all>
GET  /detect/modes      — list available modes
```

* `all` runs **PPE + Pose + Fire** as a bundle (sound is dedicated)
* `sound` is its own mode — anomaly sound runs on the audio track only

### Legacy dedicated endpoints (kept for backward compat)

```
POST /detect/ppe
POST /detect/pose
POST /detect/fire
POST /detect/sound
```

### Static asset mount

```
GET  /output/<filename>     — annotated videos
                              (ppe_annotated.mp4, pose_annotated.mp4,
                               fire_annotated.mp4, sound_annotated.mp4,
                               combined_annotated.mp4, all_annotated.mp4)
```

### Health

```
GET /          — root status
GET /health    — module + endpoint inventory
```

---

## Setup & Installation

### Prerequisites
* Python 3.10+
* Node 18+ (for the frontend)
* (Optional) NVIDIA GPU with CUDA for faster YOLO inference — falls back to CPU automatically
* No system ffmpeg required — `imageio-ffmpeg` ships a bundled binary used by both video utilities and audio extraction

### Backend
```bash
cd backend
pip install -r requirements.txt
```

Required model files:
* `backend/models/ppe_model.pt` — your custom-trained PPE YOLO weights (5 classes)
* `backend/models/yolov8s-pose.pt` — auto-downloaded by ultralytics on first run, or place locally
* `backend/models/model.pkl` — the pose ergonomic classifier (run `ml/pose/train_model.py` to generate)
* `backend/Sound Anomaly/audio_model.pkl` — the audio anomaly classifier

Optional models (auto-downloaded on first use):
* `yolov8s-worldv2.pt` — used by PPE for hazard prompts and as the last-resort fire fallback (~25 MB)
* `SalahALHaismawi/yolov26-fire-detection / best.pt` — primary fire model, fetched via `huggingface_hub` (~20 MB, cached locally)

### Run the server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: **http://localhost:8000/docs**

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies API calls to the backend on port 8000.

---

## Configuration Reference

All tunables can be overridden by environment variables before launching the backend.

### Fire detection
| Env var | Default | Effect |
|---|---|---|
| `FIRE_MODEL` | _(unset)_ | Path to a local YOLO `.pt` — overrides the HuggingFace download |
| `FIRE_HF_REPO` | `SalahALHaismawi/yolov26-fire-detection` | HF repo to fetch fire weights from |
| `FIRE_HF_FILENAME` | `best.pt` | Filename within the HF repo |
| `FIRE_INFER_IMGSZ` | `1280` | YOLO letterbox size; bump to `1536` for max small-fire recall |
| `FIRE_OUTPUT_MAX_SIDE` | `1280` | Cap on the longest side of the output video; `0` = preserve source |
| `FIRE_VERBOSE` | `1` | Per-detection log lines for diagnosis (set `0` to silence) |
| `FIRE_HSV_VERIFY` | `0` | Re-enables the optional HSV color gate (off by default; only useful with the YOLO-World fallback) |
| `FIRE_WORLD_MODEL` | `yolov8s-worldv2.pt` | Fallback model file when both custom and HF paths fail |

### PPE / hazard
Tunables live near the top of [backend/services/ppe_service.py](backend/services/ppe_service.py):

| Constant | Default | Effect |
|---|---|---|
| `_OCCLUSION_GRACE_FRAMES` | `25` | UNKNOWN grace after a worker disappears (~2 s) |
| `_PPE_STICKY_FRAMES` | `18` | Helmet/vest sticky-state window (~1.5 s) |
| `_HAZARD_PROMPTS` | `["falling rock", "loose brick", "debris", "fallen person", "falling object"]` | YOLO-World prompt list |
| `_HAZARD_CONF` (per class) | 0.30 / 0.55 (`fallen person`) | Per-class confidence floor |
| `_FALLEN_PERSON_MIN_AR` | `1.2` | Aspect ratio required for `fallen person` (rejects bending workers) |
| `_HAZARD_PERSISTENCE_FRAMES` | `2` | Consecutive-frame requirement before a hazard counts |
| `PPE_DISABLE_HAZARD=1` | _(env var)_ | Kill switch for the hazard module |

### Accident detection
Tunables in [backend/utils/accident_detector.py](backend/utils/accident_detector.py):

| Constant | Default | Effect |
|---|---|---|
| `BODY_AXIS_UPRIGHT_DEG` / `_FALLEN_DEG` | 70° / 30° | Vertical / horizontal thresholds |
| `FALL_WINDOW_FRAMES` | 23 (~1.5 s) | Fall must complete within this window |
| `CRUSHED_GOOD_FRAMES` / `_LOST_FRAMES` | 15 / 30 | Required pre-collapse and sustained-loss durations |
| `CRUSHED_BBOX_SHRINK_RATIO` | `0.65` | Lost-window bbox area must be ≤ 65% of baseline (kills walk-behind-occluder false positives) |
| `MOTIONLESS_DOWN_FRAMES` | 45 (~3 s) | Post-fall stillness window |

### Sound
Tunables in [backend/services/sound_service.py](backend/services/sound_service.py):

| Constant | Default | Effect |
|---|---|---|
| `_WINDOW_SEC` / `_HOP_SEC` | 3.0 / 1.0 | Sliding window over audio |
| `_PERSISTENCE_WINDOWS` | 2 | Consecutive-window requirement |
| `_ANOMALY_RATIO_THRESHOLD` | 0.05 | UNSAFE if more than 5% of windows are anomalous |
| `_LONG_EVENT_SEC` | 2.0 | Or any single event ≥2 s also flips UNSAFE |

---

## Known Limitations

* **PPE model** is best on standard high-vis colours. Workers in dark blue uniforms or non-yellow helmets get noisier detections; the sticky-PPE smoothing covers most flickers, but training on broader colour variants is the proper fix.
* **Pose ergonomic classifier** assumes a roughly front/rear camera angle. Side views and overhead cameras produce ambiguous joint angles.
* **Sound anomaly model** was pickled with sklearn 1.8.0; loading on 1.6.1 produces a `InconsistentVersionWarning` (predictions still work). Re-saving with the deployed sklearn version silences it.
* **`fire_best.pt`** at the project root is no longer auto-loaded; only the HuggingFace model is the default. To revive a custom fire model, set `FIRE_MODEL=<path>`.
* **Multi-worker scenes** with mixed PPE compliance: the global sticky-PPE smoothing is leaky (one worker's helmet keeps another's sticky window alive for ~1.5 s). For production stricter behaviour, port the IoU tracker into PPE for per-track sticky state.

---

## Roadmap

* Per-person sticky PPE via the existing IoU tracker (eliminates the multi-worker leak above)
* Cross-correlate hazard detection with accident events: a YOLO-World "falling brick" hit + a STRUCK pose event in the same frame should escalate STRUCK → CRITICAL
* Hand-label 100–200 pose frames for ground-truth training, freeing the ergonomic classifier from the auto-label rules
* Smoke-only detection (currently the fire prompts include `fire/flame/burning` only; smoke is a different visual signal worth adding as a separate class)
