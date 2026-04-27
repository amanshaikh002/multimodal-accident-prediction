# Implementation Details & Algorithmic Framework

This document outlines the algorithms, logic, and project architecture used in the Workplace Accident Prediction System in plain English. The formulas are simplified so they are easy to read in any text editor and easy to explain in a research paper. Each section answers three questions: **what** the technique is, **how** it works, and **why** it was chosen.

---

## 1. Project Overview & Comprehensive Details

### 1.1 Project Title
**Multimodal Vision Audio Framework for Workplace Accident Prediction**

### 1.2 Introduction & Core Motivation
Industrial workplaces — construction sites, manufacturing plants, warehouses — are high-risk environments prone to catastrophic accidents and long-term occupational injuries. Traditional safety monitoring relies heavily on manual supervision and post-incident analysis, which is reactive and prone to human error.

The framework shifts industrial safety from **reactive** to **proactive** by deploying a real-time, continuous monitoring system. It does five things in parallel:

1. **Enforces PPE compliance** (helmet, vest, gloves, boots) per worker.
2. **Analyses biomechanics** to flag unsafe postures (back, knee, neck risk).
3. **Detects accidents in progress** — falls, possible impacts, crushed workers, motionless-down events.
4. **Watches for fire hazards** with a dedicated fire/smoke detector.
5. **Listens for anomalous machine sounds** that indicate equipment failure.

Voice alerts close the loop by speaking the violation back to the floor in real time so corrective action can happen on the spot.

### 1.3 Detailed Technology Stack

The project uses a decoupled stack designed for high-throughput data processing and low-latency inference:

| Layer | Components |
|---|---|
| **Computer Vision** | Ultralytics YOLOv8 (custom PPE weights, YOLOv8s-pose), YOLO-World (open-vocabulary hazards), HuggingFace fire detector (`SalahALHaismawi/yolov26-fire-detection`), OpenCV |
| **ML / Ergonomics** | Scikit-Learn RandomForest (pose classifier), XGBoost optional, Imbalanced-Learn (SMOTE) |
| **Audio** | librosa (MFCC), bundled imageio-ffmpeg, RandomForest audio classifier |
| **Backend** | FastAPI + Uvicorn, modular routing (`ppe`, `pose`, `fire`, `sound`, `combined`, `all`) |
| **Frontend** | React 19 + Vite, per-module analysis panels, SpeechSynthesis voice alerts |
| **Infra** | huggingface_hub for model downloads, imageio_ffmpeg for portable audio decoding, joblib for model persistence |

### 1.4 System Architecture & Pipeline Flow

The framework is a **microservice-oriented** architecture: each detection signal is its own service module that can run alone or be combined with others. The UI never bottlenecks the inference loop because rendering and inference are decoupled by the JSON contract between FastAPI and React.

**The Data Pipeline Flow:**

1. **Ingestion:** raw video files are uploaded to FastAPI. The temp file is shared by all modules in a single request.
2. **Preprocessing:** dynamic frame striding (process every Nth frame to maintain real-time framerate) and resolution-aware inference. *Note:* fire detection passes the original frame to YOLO with `imgsz=1280` so portrait videos aren't squished into landscape.
3. **Per-module inference:**
    * **PPE service** runs custom YOLO + YOLO-World hazard pass.
    * **Pose service** runs YOLOv8s-pose + ergonomic classifier + accident detector.
    * **Fire service** runs the HuggingFace fire/smoke YOLO.
    * **Sound service** extracts audio and runs sliding-window MFCC classification.
    * **Combined service** runs PPE + Pose synchronously.
    * **All (Full Platform) mode** runs combined → fire overlay → single unified annotated output.
4. **Temporal stabilization:** majority voting, persistence windows, sticky-state smoothing — every signal that is noisy frame-to-frame goes through a smoother before user-facing output.
5. **Decision Engine:** the per-module statuses collapse into a single worst-case verdict (`CRITICAL` / `HIGH RISK` / `UNSAFE` / `MODERATE` / `SAFE`) with priority rules.
6. **Feedback & Output:**
    * Annotated video stream with bounding boxes, skeleton, accident overlays, and a global banner.
    * Comprehensive JSON: per-frame violations, accident events with timestamps, sound-anomaly timeline, fire ratio, recommendations.
    * Voice alerts via the browser's SpeechSynthesis API.

### 1.5 Hardware Optimization & Deployment Philosophy

* **Automatic CUDA fallback:** YOLO inference uses `device=0` if a GPU is detected, else falls back to CPU.
* **Bundled ffmpeg:** `imageio-ffmpeg` ships a portable ffmpeg binary used by both video utilities and audio extraction. **No system PATH dependency** — works on Windows out of the box.
* **Cached HF models:** `huggingface_hub` downloads weights once, then reuses the local cache.

---

## 2. Personal Protective Equipment (PPE) Detection Engine

The PPE module uses a custom YOLOv8 model trained to detect 5 classes: `human`, `helmet`, `vest`, `gloves`, `boots`. On top of raw detections, the engine applies four reliability layers: stride processing, person-centric containment, tri-state safety logic, and sticky-state smoothing. Open-vocabulary hazard detection (debris, falling rock, fallen person, etc.) runs on the same frame as a parallel signal.

### 2.1 High-Speed Frame Skipping

* **What:** process every 2nd frame instead of every frame.
* **How:** `Processed Frame = Frame Index modulo 2 == 0`. Each kept frame is also resized to 640×480 before YOLO inference.
* **Why:** running at the source 30 FPS is too slow for real-time alerts. Stride=2 halves the workload while keeping enough temporal resolution for any safety event lasting more than a third of a second.

### 2.2 PPE Compliance Logic — Center-Point Containment

* **What:** an algorithm that proves a piece of safety gear belongs to a specific worker.
* **How:**
    1. For every detected `Helmet` (or `Vest`), compute its bbox center `(x_center, y_center)`.
    2. For each `Human` bounding box, check whether that center lies inside the human bbox.
    3. The worker is compliant if **both** helmet and vest centers fall inside their bbox.
* **Why:** a helmet bbox covers ~5 % of a person's bbox area, so IoU is always below any reasonable threshold and would reject true matches. Center-point containment correctly binds small items to large containers without the IoU calibration headache.

### 2.3 Tri-State Safety Verdict (SAFE / UNSAFE / UNKNOWN)

The previous binary `safe / unsafe` contract had a critical flaw: when YOLO failed to detect a worker (occlusion, bad lighting, partial visibility), the frame was reported `SAFE` because there was "no one to check". This produced false-safe verdicts during the very moments when something might be going wrong.

* **What:** three-valued safety state per frame.
* **How (decision table):**

| Scene state | Verdict | Reason tag |
|---|---|---|
| Worker visible + helmet + vest | `SAFE` | — |
| Worker visible, missing helmet/vest | `UNSAFE` | `missing_ppe` |
| No worker, no recent worker, no motion | `SAFE` | — |
| No worker but motion present | `UNKNOWN` | `no_person_but_motion` |
| Worker seen <1.5 s ago but now hidden | `UNKNOWN` | `person_recently_seen` |

* **How (motion score):**
  ```
  motion = changed_pixel_fraction(prev_gray, curr_gray)
  // grayscale frame difference → blur → binary threshold → ratio of 1s
  motion_threshold_default = 0.015   (1.5 % of pixels)
  ```
* **How (rolling person memory):** track `frames_since_person_seen` and `ever_seen_person`. If a worker was visible within `_OCCLUSION_GRACE_FRAMES` (default 25 ≈ 2 s), treat the next frames as `UNKNOWN`, not `SAFE`.
* **Why:** absence of evidence is not evidence of absence. A scene with motion but no detected worker is suspicious — could be falling debris, a worker hidden behind machinery, or a worker who has just been struck. `UNKNOWN` is the honest verdict.

### 2.4 Sticky-PPE Smoothing

The trained PPE YOLO has poor recall on bent / crouched / non-frontal poses, on blue helmets vs. yellow ones, and on dark uniforms vs. classic high-vis vests. Without smoothing, a single bad-detection frame would flip the per-person box to `UNSAFE` even when the worker is clearly compliant for 95 % of the video.

* **What:** if a required item (helmet / vest) was visible **anywhere** in the scene within the last `_PPE_STICKY_FRAMES` frames, treat it as still worn.
* **How:**
    1. Track `last_seen_frame[helmet]` and `last_seen_frame[vest]`.
    2. Whenever the model detects the item (anywhere on screen), refresh that timer.
    3. When the per-person check fails, consult the timer: if `current_frame - last_seen ≤ window`, override `helmet_ok` / `vest_ok` to `True`.
* **Default window:** 18 processed frames ≈ 1.5 s at stride 2 / 25 fps.
* **Why:** detection misses are usually momentary (bend / turn / occlusion). A sticky window of ~1.5 s eliminates the spurious flicker without giving a free pass long enough to mask a worker who actually removed their helmet — that change would surface within the grace window.
* **Tradeoff (acknowledged):** the state is **global per video**, not per-worker. In multi-worker scenes, one worker's helmet keeps another's sticky window alive briefly. Per-track sticky state is the proper fix and is on the roadmap.

### 2.5 Open-Vocabulary Hazard Detection (YOLO-World)

The PPE pipeline can flag a worker missing gear, but it can't see falling bricks, debris piles, or a fallen person on the ground. Open-vocabulary detection plugs this hole without retraining.

* **What:** run YOLO-World (`yolov8s-worldv2.pt`) alongside the PPE pass on every processed frame, prompted with a small fixed vocabulary.
* **How:**
    1. Default prompts: `["falling rock", "loose brick", "debris", "fallen person", "falling object"]`.
    2. YOLO-World accepts text prompts via `model.set_classes(prompts)` and returns bbox detections in those classes — no fine-tuning required.
    3. A confirmed hazard immediately overrides the PPE verdict to `UNSAFE` with reason `hazard_detected` (or `missing_ppe_and_hazard`).
* **Why:** a worker in full PPE next to a falling brick is still in danger. Hazards are top priority and should never be masked by PPE compliance.

### 2.6 Hazard False-Positive Suppression (three filters)

Open-vocabulary detection on abstract concepts is noisy out of the box. The naive approach (every YOLO-World hit triggers an alert) produced ground-reflection-as-fire and bending-worker-as-fallen-person false positives. Three filters operate in series:

1. **Per-class confidence floor:**
    * `falling rock`, `loose brick`, `debris`: 0.30
    * `falling object`: 0.35
    * **`fallen person`: 0.55** — the highest because this is the label most prone to grounding on bending workers.
2. **Aspect-ratio gate (for `fallen person` only):** the bbox must satisfy `width / height ≥ 1.2` (lying horizontal). Bending and crouching workers stay vertical and are rejected by this rule alone — no extra training data needed.
3. **Temporal persistence:** the same hazard label must appear in **2 consecutive processed frames** before being acted on. A flicker is dropped silently.

* **Why:** these three filters compose multiplicatively. Even a noisy open-vocab model becomes trustworthy when each filter eliminates a different failure mode (low-conf hits, vertical-bending workers, single-frame flicker).

### 2.7 Output Schema

```json
{
  "total_frames":     int,
  "safe_frames":      int,
  "unsafe_frames":    int,
  "unknown_frames":   int,    // tri-state count (occlusion / motion-with-no-person)
  "hazard_frames":    int,    // frames that fired a confirmed hazard
  "compliance_score": float,
  "violations": [
    {"frame": int, "type": "missing_ppe" | "occlusion" | "hazard",
     "missing": [str], "hazards": [str], "reason": str}
  ]
}
```

---

## 3. Worker Pose & Ergonomics Pipeline

The pose pipeline analyses skeletal mechanics extracted from `yolov8s-pose.pt` (17 COCO keypoints). On top of raw keypoints, it runs **two parallel signals** that answer fundamentally different questions:

* **A. Ergonomic posture classifier** — *"is the worker doing their job in a way that won't injure them long-term?"*
* **B. Accident / fall detector** — *"did something just happen to this worker right now?"*

These signals do not share a model. Trying to teach a single classifier both questions is the wrong approach because the time scales (slow ergonomic patterns vs. sudden events) and the supervision are completely different.

### 3.1 Spatial Normalization (Camera-Distance Independence)

A worker walking closer to the camera produces larger pixel measurements. We solve this by converting absolute pixels into body-relative coordinates.

* **Step 1 (origin translation):** make the hip the new origin (0, 0).
    * `Relative X = Joint X - Hip X`
    * `Relative Y = Joint Y - Hip Y`
* **Step 2 (torso scaling):** use the shoulder-to-hip distance as the ruler.
    * `Torso Length = sqrt( (Shoulder X - Hip X)^2 + (Shoulder Y - Hip Y)^2 )`
* **Step 3 (normalization):**
    * `Normalized Joint X = Relative X / Torso Length`
    * `Normalized Joint Y = Relative Y / Torso Length`

**Result:** a worker bending over 10 ft from the camera produces the same normalized values as the same worker bending over 2 ft away.

### 3.2 Vector-Based Vertex Angles

Calculating joint angles using simple 2D slopes fails when the camera is at an angle. Instead, we use vector dot products to compute the true internal angle of a joint.

* **Logic:** for the knee, draw an invisible line from knee to hip and another from knee to ankle. Measure the angle between them.
* **Convention:** 180° = perfectly straight; lower = more bent.
* **Formula:** `Angle = arccos( DotProduct(V1, V2) / ( |V1| · |V2| ) )`

Three core angles are tracked:
1. **Back Angle:** vertex at the hip, between the shoulder-hip line and the hip-knee line.
2. **Knee Angle:** vertex at the knee, between the hip-knee line and the knee-ankle line.
3. **Neck Angle:** vertex at the shoulder, between the head-shoulder line and the shoulder-hip line.

### 3.3 Temporal Kinematics (Velocity & Acceleration)

To detect jerky, dangerous movements we calculate how fast each angle is changing over a rolling 8-frame window.

* **Velocity:** `Velocity = Current Angle - Previous Angle`
* **Acceleration:** `Acceleration = Current Velocity - Previous Velocity`

These features are also used by the accident detector for impact (`STRUCK`) classification.

---

## 4. Per-Person IoU Tracker

The temporal features above (velocity, acceleration, sticky state, accident-event state) only make sense when computed for a **single specific worker over time**. The previous code re-picked the "primary person" every frame using a largest-bbox heuristic, which meant the tracking buffer would silently switch between workers when the pick changed mid-clip — corrupting every time-series feature it touched.

* **What:** a lightweight greedy-IoU tracker that assigns a stable integer ID to each person across frames.
* **How:**
    1. For each new frame's detections, build an IoU matrix against all currently active tracks.
    2. Greedily match the highest-IoU pair above `iou_thresh` (default 0.30), then the next highest, until no more matches above threshold.
    3. Unmatched detections become new tracks. Unmatched tracks accumulate a `misses` counter; tracks aged past `max_age_frames` (default 15 ≈ 1 s) are retired.
    4. Each track keeps a rolling history (default 60 frames ≈ 4 s) of `(bbox, kps_xy, kps_conf, frame_idx)`.
* **Why:** simpler than DeepSORT, no Kalman filter, no embeddings — for workplace cameras (1–5 people, mostly stationary backgrounds) greedy IoU is more than enough. The downstream accident detector reads from `track.history` directly so it always sees one worker's continuous sequence, never a Frankenstein mix.

---

## 5. Accident & Fall Detection Engine

A separate, rule-based event detector layered on top of the pose keypoints. Five distinct events with their own pose signatures, time scales, and severity levels:

| Event | Trigger | Severity |
|---|---|---|
| `FALL` | body axis flips from vertical (≥ 70°) to horizontal (≤ 30°) **AND** bbox aspect ratio drops by ≥ 50 % within ~1.5 s | `CRITICAL` |
| `STRUCK` | sudden COM acceleration spike (≥ 3-σ over rolling 30-frame baseline) while the worker is still upright | `WARN` |
| `CRUSHED` | tracked at avg conf ≥ 0.60 for 15 frames, then collapses to ≤ 0.20 for 30 frames **AND** bbox shrinks to ≤ 65 % of baseline | `CRITICAL` |
| `MOTIONLESS_DOWN` | after a `FALL`, body stays horizontal with near-zero COM motion for ≥ 3 s | `CRITICAL` (medical) |
| `STUMBLE` | hip-Y drops fast then recovers; body axis never fully horizontalises | `WARN` |

### 5.1 Body-Axis Angle (the cleanest fall signal)

* **What:** the angle of the torso axis (hip → shoulder vector) from the horizontal.
* **How:**
    ```
    sm = midpoint(left_shoulder, right_shoulder)
    hm = midpoint(left_hip,      right_hip)
    dy = abs(sm.y - hm.y)        // vertical extent
    dx = abs(sm.x - hm.x)        // horizontal extent
    angle = arctan2(dy, dx) in degrees
    // 90° = perfectly vertical (upright)
    // 0°  = perfectly horizontal (lying)
    ```
* **Why:** more robust than pixel-position heuristics. A worker far from the camera produces a small body axis vector but the *angle* is still 90°. A worker who has fallen produces a small angle regardless of where in frame they landed.

### 5.2 Bounding-Box Aspect Ratio

* **What:** `h / w` of the person's bbox.
* **How:** standing person ≈ 2.0–3.0; bent forward ≈ 1.0; lying horizontal ≈ 0.3–0.6.
* **Why:** the bbox shape is a global summary that captures pose change even when individual keypoints are noisy or partially occluded. Used as the second signal in the FALL rule and as the discriminator for CRUSHED.

### 5.3 Center-of-Mass Velocity & Acceleration

* **What:** pixel-rate of change of the hip midpoint, normalized by the frame diagonal so it's resolution-independent.
* **How (3-σ spike detector for STRUCK):**
    ```
    v_t = hypot(com_t - com_{t-1}) / frame_diag
    baseline_window = recent 30 v_t values, EXCLUDING the most recent few
    z = (current_v - mean(baseline)) / (std(baseline) + 1e-6)
    if z >= 3.0 AND current_v >= 0.02 → STRUCK
    ```
* **Why:** humans walk smoothly; sudden acceleration spikes correspond to impacts. The baseline excludes the latest samples so the spike itself doesn't pollute the mean / std comparison.

### 5.4 CRUSHED Discriminator (and the false-positive lesson)

The first version of CRUSHED required only `lost_avg_conf < 0.25` for 6 frames (~0.4 s). On real warehouse footage it false-fired constantly because workers walking behind partial occlusions trip the same low-confidence pattern.

* **Fix 1 (longer windows):** `lost_frames` = 30 (~2 s) instead of 6. A real crushed worker stays partially visible for seconds, not fractions.
* **Fix 2 (bbox shrink check):** the lost-window's average bbox area must be ≤ 65 % of the baseline window's average. A walking-behind-occluder keeps a similar bbox; a trapped worker only shows a portion.
* **Why both:** confidence alone is too noisy. Adding a geometric constraint (bbox area) eliminates the most common false-positive pattern (transient occlusion) without requiring more training data.

### 5.5 Cooldowns & Priority

Each event type has its own cooldown window so the detector doesn't spam the same event every frame:

```
FALL:             45 frames (~3 s)
STRUCK:           15 frames
CRUSHED:          60 frames
MOTIONLESS_DOWN:  90 frames
STUMBLE:          15 frames
```

Within a single frame, only **one event** is emitted per track, picked by priority: `CRUSHED > MOTIONLESS_DOWN > FALL > STRUCK > STUMBLE`. Additionally, `STUMBLE` is suppressed during a recent FALL cooldown — the dropping hip is part of the same fall, not a separate event.

### 5.6 Overlay Renderer with TTL

Once an event fires, it should remain visible on the output video for several seconds so a viewer doesn't miss it.

* **What:** a stateful overlay layer that keeps active events on screen for `ttl_frames` (default 45 ≈ 3 s).
* **How:**
    1. When the same `(track_id, type)` re-fires, the existing entry's TTL is **refreshed** rather than appended — no piling-up duplicates.
    2. CRITICAL events draw a pulsing thick red border (alternating thickness every 4 frames); WARN draws a solid orange border.
    3. The top banner names the active events ("WORKER FALL" for single events, "FALL | STRUCK" compact form for multiple).
    4. Each affected worker's bbox is highlighted in the event color with a per-bbox tag ("WORKER FALL conf=0.69").
    5. Banner / tag text scales with frame height so 1080p+ output stays readable.
* **Why:** raw flicker is unprofessional and easy to miss. TTL persistence + visual hierarchy (color, pulse) makes the alert impossible to overlook even on a quick rewatch.

The renderer survives no-pose frames too: if YOLO momentarily loses keypoints in the middle of a fall, the red border + banner stay on screen because the renderer is called on every output frame regardless of pose state.

---

## 6. Fuzzy Auto-Labeling Algorithm (Pose Training Data)

To train the pose ergonomic model, thousands of frames must be labelled `SAFE` / `MODERATE` / `UNSAFE`. Hard cutoffs (e.g. "exactly 140° is unsafe") would teach the model an artificial step function. Instead, a fuzzy-logic scoring system produces continuous risk scores that the classifier learns to reproduce.

### 6.1 Risk Scoring (Sigmoid Functions)

Each angle is graded on a continuous 0.0–1.0 scale using an inverted sigmoid:
* **Back Risk:** centered at 155°. 180° → ~0.0; 120° → ~0.9.
* **Knee Risk:** centered at 145°.
* **Neck Risk:** centered at 155°.

### 6.2 Weighted Classification

```
Total Score = (0.50 * Back Risk) + (0.30 * Knee Risk) + (0.20 * Neck Risk)

SAFE:     Total Score < 0.30
MODERATE: 0.30 ≤ Total Score < 0.55
UNSAFE:   Total Score ≥ 0.55
```

To prevent the model from memorising the exact formula, **5 % of the labels are flipped randomly** (label noise injection). This forces the model to learn the overall pattern rather than a specific cutoff.

---

## 7. Machine Learning Training Pipeline (Pose Classifier)

A RandomForest classifier (`ml/pose/model.pkl`) is trained on the 9-feature vector: `back_angle, knee_angle, neck_angle, norm_shoulder_x/y, norm_hip_x/y, norm_knee_x/y`.

### 7.1 Handling Dataset Imbalance (SMOTE)

In real footage, workers are usually either standing safely or bending unsafely; the `MODERATE` transition is rare.

* **SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic `MODERATE` examples by interpolating between existing ones.
* **Class weighting** during training penalises errors on the rare class more heavily.
* **Why:** without SMOTE, the model learns a `safe / unsafe` binary and treats `MODERATE` as noise.

### 7.2 Data Leakage Prevention (Video-Based Split)

Frames within a video are highly correlated. A random frame-level train/test split would let the model memorise specific workers and lighting conditions rather than the underlying biomechanics.

* **Fix:** entire videos are placed in either the train or test set — never split.

---

## 8. Real-Time Inference & Hybrid Fallback Engine (Pose)

When the system runs live, the pose classifier's prediction goes through a multi-stage safety filter to prevent false alarms:

1. **Geometric rule overrides (highest priority):**
    * `Back Angle < 110°` (worker bent in half) → force `UNSAFE`.
    * `Back Velocity < -12.0` (bent down very fast) → force `UNSAFE`.
    * `Back Angle > 150° AND 70° ≤ Knee Angle ≤ 130°` (proper squat) → force `SAFE`.
    * Why: rules are fast, deterministic, and correct for extreme cases. ML is reserved for the ambiguous middle.
2. **ML probability thresholds:**
    * `prob_unsafe > 0.80` → `UNSAFE`.
    * `prob_safe > 0.55` → `SAFE`.
    * Otherwise → `MODERATE`.
3. **Confidence-based downgrading:** if `pred == UNSAFE` but average keypoint confidence < 0.55, downgrade to `MODERATE` — don't trust an unsafe verdict on noisy keypoints.
4. **Temporal majority voting:** the last 5 frame predictions are stored in a deque; the most frequent label wins. Eliminates UI flicker.
5. **Consecutive-frame violation gate:** a violation is only logged after 3+ consecutive `UNSAFE` frames. Suppresses transient blips.

---

## 9. Fire Hazard Detection Engine

The fire module was rebuilt from the ground up after the original `fire_best.pt` was found to miss obvious fires. The new pipeline uses a HuggingFace fire/smoke model with three reliability layers.

### 9.1 Model Loader Priority

```
1. FIRE_MODEL env var        — local custom YOLO weights (overrides everything else)
2. HuggingFace fire model    — SalahALHaismawi/yolov26-fire-detection (default,
                                 ~20 MB, classes: fire, other, smoke)
3. YOLO-World fallback       — yolov8s-worldv2.pt prompted with fire vocabulary
                                 (only if both above fail)
```

* **Why three tiers:** the production default uses a real fire-trained model that lands detections on actual flames. The custom-path option lets the user drop in their own weights without code changes. The YOLO-World fallback is the offline-resilient backstop.

### 9.2 Bundled ffmpeg via imageio_ffmpeg

* **What:** instead of relying on system PATH, the fire service uses `imageio_ffmpeg.get_ffmpeg_exe()` for any ffmpeg subprocess work.
* **Why:** Windows doesn't ship ffmpeg by default, and Python's subprocess doesn't see user-installed binaries on PATH unless the shell environment is exactly right. The `imageio-ffmpeg` Python package ships a portable binary, so this works on every platform out of the box.

### 9.3 Resolution & Aspect-Ratio Handling

The original code resized every frame to 640×480 *before* YOLO inference. For a portrait video like 1080×1920, that turned the worker into a horizontally-squished blob — and the model had to identify fire on top of a deformed image.

* **Fix 1:** pass the **original frame** to YOLO with `imgsz=1280`. Ultralytics letterboxes internally to `1280×1280` while preserving aspect ratio, then maps detections back to the original frame coordinates.
* **Fix 2:** output video keeps the source aspect ratio, capped at `FIRE_OUTPUT_MAX_SIDE=1280` so 4K input doesn't write multi-GB files.
* **Fix 3:** banner / box / font sizes scale with frame height (`max(1.0, h / 720.0)`) so annotations stay legible on every resolution.
* **Why:** the model's recall depends entirely on seeing fire pixels in their natural shape and resolution. Squishing kills that.

### 9.4 Temporal Persistence (2 consecutive frames)

A frame counts as fire-positive only after the raw detector has fired on **2 consecutive** processed frames. Single-frame flicker is dropped silently. **Why:** a true fire persists for many seconds; a momentary high-conf hit on a bright object (lamp, sunset, headlight) does not.

### 9.5 Verbose Per-Detection Logging (default ON)

```
[FIRE] frame=42  HIT       label='fire'   conf=0.713
[FIRE] frame=42  OTHER     label='smoke'  conf=0.412
[FIRE] frame=58  LOW_CONF  label='fire'   conf=0.143
[FIRE] frame=120 EMPTY     (model produced 0 raw detections)
```

* **Why:** when the model misses an obvious fire, we want to know exactly why — too low confidence? wrong class? nothing detected at all? The default-ON verbose logging surfaces this transparently. Set `FIRE_VERBOSE=0` to silence.

### 9.6 `draw_banner=False` for Chained Use

The fire service normally draws a `FIRE DETECTED` / `NO FIRE DETECTED` banner at the top of every frame. When chained inside the Full Platform pipeline, this banner would paint over the upstream combined-mode banner.

* **Fix:** `process_fire_video(..., draw_banner=False)` skips the banner draw call but still draws the per-bbox fire highlight.
* **Why:** keeps both signals visible without UI conflict. Used by the Full Platform "all" pipeline.

### 9.7 Graceful Failure

Every audio / codec / file error is converted into a graceful 200 response with a clear message — never a 4xx. The same philosophy applies to the sound module (see §10). **Why:** "no analysis possible" is a valid diagnostic outcome, not an HTTP error.

---

## 10. Anomaly Sound Detection Engine

A sliding-window machine-sound classifier that listens for grinding, hissing, banging, and other abnormal mechanical sounds that may indicate equipment failure. Trained on standard machine-sound datasets.

### 10.1 Audio Extraction

* **Primary path:** `librosa.load(video_path, sr=22050, mono=True)` — works for most mp4 builds via the soundfile / audioread backends.
* **Fallback path:** if librosa returns empty or raises, transcode with the bundled `imageio_ffmpeg` binary into a temp `.wav`, then load with librosa.
* **Why two paths:** librosa's direct load is fastest when it works, but some codec / container combinations break it silently. The ffmpeg fallback is universal.

### 10.2 Sliding-Window MFCC + RandomForest

* **Window:** 3 seconds (matches the model's training duration).
* **Hop:** 1 second (one prediction per second of audio).
* **Feature:** 13-dim MFCC mean (`np.mean(librosa.feature.mfcc(y, sr=sr, n_mfcc=13).T, axis=0)`) — the same feature pipeline the model was trained on.
* **Classifier:** RandomForest in `backend/Sound Anomaly/audio_model.pkl`. Returns `predict` (0 / 1) and `predict_proba` per window.

### 10.3 Two-Window Persistence

* **What:** a window only counts as anomalous if it is part of a run of 2+ consecutive raw-anomaly windows.
* **Why:** single-window false positives (a brief loud bang, a passing forklift) are common and do not indicate equipment failure. Real abnormal sounds (a stuck bearing, a ruptured hose) persist for seconds.

### 10.4 Event Grouping

Confirmed anomalous windows are collapsed into contiguous events with `start_sec`, `end_sec`, `duration_sec`, `avg_confidence`, `max_confidence`, `max_anomaly_prob`. The frontend `SoundPanel` renders these as a horizontal timeline with red bands per event.

### 10.5 Verdict

```
UNSAFE if:
  anomaly_ratio > 5%   (more than 5% of windows are anomalous)
  OR any single confirmed event >= 2 seconds long
SAFE otherwise.
```

### 10.6 Graceful Failure (no audio track)

Any extraction failure (no audio track, codec issue, ffmpeg missing, corrupt file) produces a 200 response with `status: SAFE` and a clear message naming the actual error class. **Never a 4xx.** The previous version returned 422 on no-audio videos, which made the frontend look broken when the real problem was just a silent video.

---

## 11. Combined Multimodal Safety Engine (PPE + Pose)

The combined service runs PPE and Pose synchronously on the same frame, then merges the verdicts.

### 11.1 Synchronous Multi-Model Inference

* The frame is passed through `ppe_model.pt` and `yolov8s-pose.pt` in the same loop iteration — no double video read.
* Bounding boxes from both models share the same coordinate space.
* Each detected person gets:
    * **Pose label** (`SAFE` / `MODERATE` / `UNSAFE`) from the ergonomic classifier with 5-frame majority vote.
    * **PPE check** (helmet + vest center-point containment).
    * **Combined per-person status:** `HIGH RISK` if both unsafe, `UNSAFE` if either, `MODERATE` if pose only, else `SAFE`.

### 11.2 Sticky-PPE in Combined Mode

The same sticky-state smoothing from §2.4 is also active here. A bend / brief detection miss does not flip the per-person box to `UNSAFE` if the item was visible in the recent window. **Why:** consistent UX — combined mode and standalone PPE mode produce the same per-frame verdict on the same input.

### 11.3 Clutter-Free Visualization

* Skeletal mesh drawn first (clean, no per-joint labels).
* Single color-coded bounding box per person — color tells the safety state at a glance.
* Per-person PPE status icons inside the top edge of the bounding box.
* Global banner at the top of the frame names the worst current status across all visible workers.
* Accident overlays (red border + banner from §5.6) sit on top.

---

## 12. Full Platform Mode (`mode=all`)

The "all" route runs **PPE + Pose + Fire** as a unified pipeline. **Sound is intentionally not included** — it has its own dedicated mode because audio analysis is independent of video frame timing.

### 12.1 Pipeline Stages

```
1. process_combined_video(input → combined_annotated.mp4)
     ↳ skeleton + per-person box + accident overlays + global banner
2. process_fire_video(combined_annotated.mp4 → all_annotated.mp4,
                      draw_banner=False)
     ↳ overlays fire bounding boxes on top of the combined video
3. Output: all_annotated.mp4   (PPE + Pose + Accident + Fire annotations
                                in a single file)
```

### 12.2 Why this chained approach

* **Single output file** — the user sees one video with everything, not three separate ones to switch between.
* **Banner conflict avoided** via `draw_banner=False` on the fire pass.
* **Each module's full pipeline is reused** — no duplicated annotation code, no risk of drift between standalone and combined visualization.

---

## 13. Unified Decision Engine

The combined verdict across all signals follows this priority hierarchy:

```
CRITICAL    fire UNSAFE  OR  accident_status == CRITICAL
HIGH RISK   PPE UNSAFE  AND  Pose UNSAFE
UNSAFE      PPE UNSAFE  OR  Pose UNSAFE  OR  accident_status == WARN
            OR  sound UNSAFE
MODERATE    Pose MODERATE only
SAFE        all clear
```

### 13.1 CRITICAL is not always fire

The previous frontend hardcoded "CRITICAL — FIRE HAZARD" as the banner label whenever `final_status === "CRITICAL"`. After the accident detector was added, this produced misleading alerts: a worker fall would correctly trigger CRITICAL but the banner would still say FIRE HAZARD even when fire was 0 %.

* **Fix:** the frontend `AllPanel` now inspects `fire_status` and `accident_status` to label the CRITICAL banner correctly:
    * Fire only → "CRITICAL — FIRE HAZARD"
    * Accident only → "CRITICAL — WORKER ACCIDENT" + the most-urgent event type ("Worker is down and not moving", "Worker may be trapped or covered", "Worker fall", etc.)
    * Both → "CRITICAL — FIRE & WORKER ACCIDENT"
* **Why:** the user must immediately know what kind of emergency they're looking at — fire and accident require completely different responses.

### 13.2 Backward Compatibility

`get_final_status` accepts `accident_status` and `sound_status` as optional 4th and 5th arguments, defaulting to `"SAFE"`. Existing 3-arg callers continue to work unchanged — the new signals only escalate the verdict, never lower it.

---

## 14. Frontend Dashboard

A React 19 + Vite single-page dashboard. The module dropdown drives the pipeline; the right column adapts to whichever mode was last analysed.

### 14.1 Per-Module Analysis Panels

| Component | Mode | What it shows |
|---|---|---|
| `AnalysisPanel` | `ppe`, `pose` | Compliance / safety score, per-frame violations |
| `CombinedPanel` | `combined` | Merged PPE + Pose metrics |
| `FirePanel` | `fire` | Fire status, ratio, frame counts |
| `SoundPanel` | `sound` | Status banner, anomaly ratio bar, **horizontal timeline** with red bands per event, scrollable event list with timestamps + confidences |
| `AllPanel` | `all` | Full Platform summary, dynamic CRITICAL labelling, optional **Accident Events** card when accident events fired |
| `ViolationsList` | all | Frame-by-frame violations from any module |
| `Recommendations` | all | Actionable text per detected issue |

### 14.2 Audio Alerts (the generic-pose-alert lesson)

The original alert system had activity-specific phrasing:
* `"Alert! Unsafe lifting posture detected."` (when reason mentioned back / lifting)
* `"Warning! Unsafe knee posture detected."` (when reason mentioned knee)
* `"Warning! Unsafe neck posture detected."` (when reason mentioned neck)

This produced **false claims** — videos with no lifting at all triggered the lifting alert. The pose model only knows the body angles are outside the safe range; it doesn't know whether the worker is lifting, walking, reaching, or just bending to pick up a tool.

* **Fix:** all three branches collapsed into one generic alert:
  > *"Warning! Unsafe body posture detected. Review worker form."*
* The regex now matches any pose-related keyword (`back|posture|lifting|knee|stiff|leg|neck|bend`) but they all map to the same single message, fired at most once per analysis run.
* **Why:** speak only what the model can actually verify. PPE alerts (helmet / vest / gloves / boots) stay specific because those are deterministic visual signals.

### 14.3 Voice Alert Stack

```
Fire UNSAFE         "Warning! Fire detected. Evacuate immediately and call emergency services."
Sound UNSAFE        "Warning! Anomalous machine sounds detected. Inspect the equipment immediately."
Accident CRITICAL   "Warning! A worker accident has been detected. Dispatch first aid immediately."
PPE missing helmet  "Warning! Worker is not wearing a safety helmet."
PPE missing vest    "Alert! Worker is missing a high-visibility vest."
Pose violation      "Warning! Unsafe body posture detected. Review worker form."
PPE + Pose combo    "Warning! Worker is unsafe due to missing PPE and unsafe posture."
```

Alerts deduplicate within a session — the same exact message is not spoken twice.

### 14.4 SoundPanel Timeline Visualization

For sound mode, the panel shows a horizontal bar representing the audio duration with red bands at the timestamps of each anomalous event. Tooltips show start/end time and max confidence. Below the bar, a chronological list shows every event's full details. **Why:** sound events have natural timestamps, and a visual timeline is the most intuitive way to surface "this happened from 4 s to 7 s" — much more readable than a JSON dump.

---

## 15. Configuration Reference (Tunables)

All tunables can be overridden via environment variables before launching the backend.

### 15.1 Fire Module
| Env var | Default | Effect |
|---|---|---|
| `FIRE_MODEL` | (unset) | Path to a local YOLO `.pt` — overrides the HuggingFace download |
| `FIRE_HF_REPO` | `SalahALHaismawi/yolov26-fire-detection` | HF repo to fetch weights from |
| `FIRE_HF_FILENAME` | `best.pt` | Filename within the HF repo |
| `FIRE_INFER_IMGSZ` | `1280` | YOLO letterbox size |
| `FIRE_OUTPUT_MAX_SIDE` | `1280` | Cap on output longest side (`0` = preserve source) |
| `FIRE_VERBOSE` | `1` | Per-detection log lines |
| `FIRE_HSV_VERIFY` | `0` | Re-enable optional HSV color gate |

### 15.2 PPE / Hazard Module (in `ppe_service.py`)
| Constant | Default | Effect |
|---|---|---|
| `_OCCLUSION_GRACE_FRAMES` | 25 | UNKNOWN window after a worker disappears (~2 s) |
| `_PPE_STICKY_FRAMES` | 18 | Helmet/vest sticky-state window (~1.5 s) |
| `_HAZARD_PROMPTS` | 5-prompt list | YOLO-World hazard vocabulary |
| `_HAZARD_CONF[fallen person]` | 0.55 | Highest per-class floor |
| `_FALLEN_PERSON_MIN_AR` | 1.2 | Aspect ratio required for `fallen person` |
| `_HAZARD_PERSISTENCE_FRAMES` | 2 | Consecutive-frame requirement |
| `PPE_DISABLE_HAZARD=1` | (env var) | Kill switch for the hazard module |

### 15.3 Accident Module (in `accident_detector.py`)
| Constant | Default | Effect |
|---|---|---|
| `BODY_AXIS_UPRIGHT_DEG` / `_FALLEN_DEG` | 70° / 30° | Vertical / horizontal thresholds |
| `FALL_WINDOW_FRAMES` | 23 (~1.5 s) | Fall must complete within this window |
| `CRUSHED_GOOD_FRAMES` / `_LOST_FRAMES` | 15 / 30 | Pre-collapse and sustained-loss durations |
| `CRUSHED_BBOX_SHRINK_RATIO` | 0.65 | Lost-window bbox ≤ 65 % of baseline |
| `MOTIONLESS_DOWN_FRAMES` | 45 (~3 s) | Post-fall stillness window |

### 15.4 Sound Module (in `sound_service.py`)
| Constant | Default | Effect |
|---|---|---|
| `_WINDOW_SEC` / `_HOP_SEC` | 3.0 / 1.0 | Sliding window over audio |
| `_PERSISTENCE_WINDOWS` | 2 | Consecutive-window requirement |
| `_ANOMALY_RATIO_THRESHOLD` | 0.05 | UNSAFE if > 5 % of windows are anomalous |
| `_LONG_EVENT_SEC` | 2.0 | Or any single event ≥ 2 s also flips UNSAFE |

---

## 16. Known Limitations & Roadmap

### 16.1 Known Limitations

* **PPE model recall** is best on standard high-vis colours. Workers in dark blue uniforms or non-yellow helmets get noisier detections; sticky-PPE smoothing covers most flickers, but training on broader colour variants is the proper fix.
* **Pose ergonomic classifier** assumes a roughly front/rear camera angle. Side views and overhead cameras produce ambiguous joint angles.
* **Sound model sklearn version skew**: pickled with sklearn 1.8.0; loading on 1.6.1 produces a warning (predictions still work). Re-saving with the deployed version silences it.
* **Multi-worker sticky-PPE leak**: state is global per video, not per-track. One worker's helmet keeps another worker's sticky window alive briefly. Per-track sticky state is the proper fix.
* **`fire_best.pt`** at the project root is no longer auto-loaded; only the HuggingFace model is the default. To revive a custom fire model, set `FIRE_MODEL=<path>`.

### 16.2 Roadmap

* **Per-person sticky PPE** via the existing IoU tracker — eliminates the multi-worker leak.
* **Hazard ↔ accident cross-correlation** — a YOLO-World "falling brick" hit + a STRUCK pose event in the same frame should escalate STRUCK → CRITICAL.
* **Hand-labelled pose ground truth** — frees the ergonomic classifier from the auto-label rules.
* **Smoke detection** — currently the fire prompts include `fire/flame/burning` only; smoke is a different visual signal worth adding as a separate class.
* **3D pose lifting** (e.g. MediaPipe BlazePose) — would solve the side-view / overhead camera ambiguity in §16.1.
