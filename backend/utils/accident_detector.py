"""
accident_detector.py
=====================
Rule-based accident & fall detection layered on top of pose keypoints.

Detects, per tracked person, the following events:

  - FALL              — body axis flips from vertical to horizontal AND the
                        bbox aspect ratio collapses, all within a short
                        window (~1.5 s). CRITICAL severity.
  - STRUCK            — sudden center-of-mass acceleration spike (≥3-sigma
                        over recent rolling baseline) while the person is
                        still upright. WARN severity.
  - CRUSHED           — keypoint confidence collapses sharply on a track
                        that was previously well-detected (worker is
                        partially visible but mostly occluded).
                        CRITICAL severity.
  - MOTIONLESS_DOWN   — after a fall, the person stays horizontal with
                        near-zero COM motion for ≥ 3 sec. CRITICAL.
  - STUMBLE           — hip-Y drops fast then recovers; body axis never
                        fully horizontalises. WARN.

The detector is rule-based on temporal patterns. It does NOT use the
ergonomic ML classifier (which is trained on static-frame angle features
and has no temporal awareness). Designed to run alongside the existing
posture classifier — they answer different questions.

All thresholds are first-pass guesses based on biomechanics. Tune from
real footage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Use a TYPE_CHECKING-free import; tracker module is sibling.
from utils.pose_tracker import TrackedPerson, TrackFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keypoint indices (COCO-17)
# ---------------------------------------------------------------------------
KP_LSHOULDER = 5
KP_RSHOULDER = 6
KP_LHIP      = 11
KP_RHIP      = 12

# A keypoint must be at least this confident to feed into geometric features.
_KP_CONF_FLOOR = 0.40


# ---------------------------------------------------------------------------
# Tunables — first-pass values, calibrate from real data.
# ---------------------------------------------------------------------------

# Body axis from horizontal: 90° = perfectly vertical, 0° = perfectly horizontal.
BODY_AXIS_UPRIGHT_DEG = 70.0   # ≥ this counts as "standing"
BODY_AXIS_FALLEN_DEG  = 30.0   # ≤ this counts as "horizontal"

# Aspect-ratio collapse: AR_after / AR_before must drop below this to count.
AR_COLLAPSE_RATIO     = 0.50

# Fall must complete within this many processed frames (≈ 1.5 s @ 15 fps).
FALL_WINDOW_FRAMES    = 23

# STRUCK: COM acceleration must exceed (mean + Z * std) of recent baseline.
STRUCK_Z_THRESHOLD    = 3.0
STRUCK_BASELINE_FRAMES = 30   # ~2 s rolling baseline
STRUCK_MIN_VEL_NORM    = 0.02   # absolute floor — below this we don't fire

# CRUSHED: avg keypoint conf collapse from "well-tracked" to "lost".
# A real "trapped/crushed" event has the worker partially visible at very
# low confidence for several SECONDS, not just half a second. The previous
# 6-frame (~0.4 s) lost window false-fired on workers walking behind boxes
# / partial occlusions in real warehouse footage. Bumped to ~2 s of
# sustained low conf with a longer "good" baseline before that.
CRUSHED_CONF_GOOD     = 0.60   # avg conf in the pre-collapse window
CRUSHED_CONF_LOST     = 0.20   # avg conf in the sustained-lost window
CRUSHED_GOOD_FRAMES   = 15     # ~1 s of confident tracking baseline
CRUSHED_LOST_FRAMES   = 30     # ~2 s of sustained partial-visibility loss
# Plus: the bbox area must shrink meaningfully -- a person walking behind
# an occluder keeps roughly the same bbox; a trapped person typically has
# only a partial body visible, so the bbox area drops.
CRUSHED_BBOX_SHRINK_RATIO = 0.65   # avg lost-bbox area must be <= 65% of avg good-bbox area

# MOTIONLESS_DOWN: post-fall stillness window.
MOTIONLESS_DOWN_FRAMES   = 45     # ≈ 3 s at 15 fps
MOTIONLESS_VEL_NORM_MAX  = 0.005  # max normalized COM velocity to count as "still"

# STUMBLE: hip-Y drops fast then recovers without going horizontal.
STUMBLE_HIP_DROP_NORM = 0.10   # ≥ 10 % of frame height drop in window
STUMBLE_RECOVERY_FRAMES = 10   # within this many frames

# Cooldowns to avoid re-firing on the same event.
EVENT_COOLDOWN_FRAMES: Dict[str, int] = {
    "FALL":             45,    # ~3 s
    "STRUCK":           15,
    "CRUSHED":          60,
    "MOTIONLESS_DOWN":  90,    # only re-fire once per long pause
    "STUMBLE":          15,
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class AccidentEvent:
    frame:      int
    track_id:   int
    type:       str        # "FALL" | "STRUCK" | "CRUSHED" | "MOTIONLESS_DOWN" | "STUMBLE"
    severity:   str        # "WARN" | "CRITICAL"
    confidence: float
    reason:     str

    def to_dict(self) -> Dict:
        return {
            "frame":      int(self.frame),
            "track_id":   int(self.track_id),
            "type":       self.type,
            "severity":   self.severity,
            "confidence": round(float(self.confidence), 3),
            "reason":     self.reason,
        }


# ---------------------------------------------------------------------------
# Per-frame geometric helpers
# ---------------------------------------------------------------------------

def _midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) * 0.5


def _shoulder_hip_midpoints(
    kps_xy: np.ndarray, kps_conf: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (shoulder_mid, hip_mid) if both shoulders + both hips are confident, else None."""
    if (kps_conf[KP_LSHOULDER] < _KP_CONF_FLOOR or
        kps_conf[KP_RSHOULDER] < _KP_CONF_FLOOR or
        kps_conf[KP_LHIP]      < _KP_CONF_FLOOR or
        kps_conf[KP_RHIP]      < _KP_CONF_FLOOR):
        return None
    sm = _midpoint(kps_xy[KP_LSHOULDER], kps_xy[KP_RSHOULDER])
    hm = _midpoint(kps_xy[KP_LHIP],      kps_xy[KP_RHIP])
    return sm, hm


def body_axis_angle_from_horizontal(
    kps_xy: np.ndarray, kps_conf: np.ndarray,
) -> Optional[float]:
    """
    Angle (deg) of the torso axis (hip→shoulder) from the horizontal.

    90° → perfectly vertical (upright)
     0° → perfectly horizontal (lying)
    Returns None when essential keypoints are too low-confidence.
    """
    mids = _shoulder_hip_midpoints(kps_xy, kps_conf)
    if mids is None:
        return None
    sm, hm = mids
    dy = abs(float(sm[1] - hm[1]))   # vertical extent
    dx = abs(float(sm[0] - hm[0]))   # horizontal extent
    if dy == 0.0 and dx == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx))


def bbox_aspect_ratio(bbox: np.ndarray) -> Optional[float]:
    """h / w of an xyxy bbox. None if degenerate."""
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if w <= 1.0 or h <= 0.0:
        return None
    return h / w


def hip_midpoint(kps_xy: np.ndarray, kps_conf: np.ndarray) -> Optional[np.ndarray]:
    if kps_conf[KP_LHIP] < _KP_CONF_FLOOR or kps_conf[KP_RHIP] < _KP_CONF_FLOOR:
        return None
    return _midpoint(kps_xy[KP_LHIP], kps_xy[KP_RHIP])


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

@dataclass
class _TrackState:
    last_event_frame: Dict[str, int] = field(default_factory=dict)
    last_fall_frame:  int = -1


class AccidentDetector:
    """
    Stateful per-track accident detector. One instance per video.

    Usage
    -----
        detector = AccidentDetector(frame_w=W, frame_h=H)
        for frame_idx, tracks in stream:
            for track in tracks:
                event = detector.evaluate(track, frame_idx)
                if event:
                    ...

    The detector keeps a tiny per-track state (event cooldowns, last-fall
    frame for motionless tracking). It does NOT keep its own copy of pose
    data — it reads from `track.history` directly.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        fall_window_frames:      int   = FALL_WINDOW_FRAMES,
        upright_deg:             float = BODY_AXIS_UPRIGHT_DEG,
        fallen_deg:              float = BODY_AXIS_FALLEN_DEG,
        ar_collapse_ratio:       float = AR_COLLAPSE_RATIO,
        struck_z_threshold:      float = STRUCK_Z_THRESHOLD,
        motionless_down_frames:  int   = MOTIONLESS_DOWN_FRAMES,
    ) -> None:
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        # Frame diagonal — used to normalize pixel velocities to a
        # resolution-independent scale.
        self.frame_diag = float(math.hypot(self.frame_w, self.frame_h))

        self.fall_window_frames     = int(fall_window_frames)
        self.upright_deg            = float(upright_deg)
        self.fallen_deg             = float(fallen_deg)
        self.ar_collapse_ratio      = float(ar_collapse_ratio)
        self.struck_z_threshold     = float(struck_z_threshold)
        self.motionless_down_frames = int(motionless_down_frames)

        self._states: Dict[int, _TrackState] = {}

    # ------------------------------------------------------------------

    def evaluate(self, track: TrackedPerson, frame_idx: int) -> Optional[AccidentEvent]:
        """
        Return the highest-priority accident event for this track at this
        frame, or None if nothing fires.

        Priority order: CRUSHED > MOTIONLESS_DOWN > FALL > STRUCK > STUMBLE.
        Only one event per track per frame.
        """
        if len(track.history) < 2:
            return None

        state = self._states.setdefault(track.track_id, _TrackState())

        # Run detectors in priority order; first non-None wins.
        for fn in (
            self._check_crushed,
            self._check_motionless_down,
            self._check_fall,
            self._check_struck,
            self._check_stumble,
        ):
            ev = fn(track, frame_idx, state)
            if ev is not None:
                self._stamp_cooldown(state, ev)
                return ev
        return None

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _check_fall(
        self, track: TrackedPerson, frame_idx: int, state: _TrackState,
    ) -> Optional[AccidentEvent]:
        if self._in_cooldown(state, "FALL", frame_idx):
            return None

        window = self._tail_window(track, self.fall_window_frames)
        if len(window) < 4:
            return None

        # Earliest "well-defined upright" reading in the window
        early_axis: Optional[float] = None
        early_ar:   Optional[float] = None
        for tf in window:
            axis = body_axis_angle_from_horizontal(tf.kps_xy, tf.kps_conf)
            ar   = bbox_aspect_ratio(tf.bbox)
            if axis is None or ar is None:
                continue
            if axis >= self.upright_deg:
                early_axis = axis
                early_ar   = ar
                break

        if early_axis is None or early_ar is None:
            return None

        # Most recent axis & AR
        latest = window[-1]
        late_axis = body_axis_angle_from_horizontal(latest.kps_xy, latest.kps_conf)
        late_ar   = bbox_aspect_ratio(latest.bbox)
        if late_axis is None or late_ar is None:
            return None

        if late_axis > self.fallen_deg:
            return None
        if late_ar > early_ar * self.ar_collapse_ratio:
            return None

        # Confidence: how decisive is the flip?
        axis_drop = (early_axis - late_axis) / 90.0           # 0..1
        ar_drop   = max(0.0, 1.0 - (late_ar / early_ar))       # 0..1
        confidence = float(np.clip(0.5 + 0.5 * axis_drop * ar_drop, 0.5, 0.99))

        state.last_fall_frame = frame_idx
        return AccidentEvent(
            frame      = frame_idx,
            track_id   = track.track_id,
            type       = "FALL",
            severity   = "CRITICAL",
            confidence = confidence,
            reason     = (
                f"Body axis {early_axis:.0f}° -> {late_axis:.0f}° and bbox "
                f"aspect ratio {early_ar:.2f} -> {late_ar:.2f} within "
                f"{len(window)} processed frames"
            ),
        )

    def _check_struck(
        self, track: TrackedPerson, frame_idx: int, state: _TrackState,
    ) -> Optional[AccidentEvent]:
        if self._in_cooldown(state, "STRUCK", frame_idx):
            return None

        # Compute COM (hip mid) velocities across the track history.
        coms = self._com_series(track)
        if len(coms) < STRUCK_BASELINE_FRAMES + 2:
            return None

        vels = self._normalized_velocities(coms)  # length = len(coms) - 1
        if len(vels) < STRUCK_BASELINE_FRAMES + 1:
            return None

        # Use prior baseline window, EXCLUDING the most recent few samples
        # so the spike itself doesn't pollute the mean / std.
        baseline = np.asarray(vels[-(STRUCK_BASELINE_FRAMES + 1):-1], dtype=np.float32)
        current  = float(vels[-1])

        if current < STRUCK_MIN_VEL_NORM:
            return None

        mu  = float(baseline.mean())
        sig = float(baseline.std() + 1e-6)
        z   = (current - mu) / sig
        if z < self.struck_z_threshold:
            return None

        # Person should still be upright — if they're already lying, this
        # is part of a fall, not a struck-while-standing impact.
        latest_axis = body_axis_angle_from_horizontal(
            track.history[-1].kps_xy, track.history[-1].kps_conf,
        )
        if latest_axis is not None and latest_axis < self.upright_deg - 10.0:
            return None

        confidence = float(np.clip(0.5 + min(z / 6.0, 0.49), 0.5, 0.99))
        return AccidentEvent(
            frame      = frame_idx,
            track_id   = track.track_id,
            type       = "STRUCK",
            severity   = "WARN",
            confidence = confidence,
            reason     = (
                f"COM velocity spike z={z:.1f}sigma "
                f"({current:.4f} vs baseline mu={mu:.4f}, sigma={sig:.4f})"
            ),
        )

    def _check_crushed(
        self, track: TrackedPerson, frame_idx: int, state: _TrackState,
    ) -> Optional[AccidentEvent]:
        if self._in_cooldown(state, "CRUSHED", frame_idx):
            return None
        need = CRUSHED_GOOD_FRAMES + CRUSHED_LOST_FRAMES
        if len(track.history) < need:
            return None

        recent = list(track.history)[-need:]
        good_window = recent[:CRUSHED_GOOD_FRAMES]
        lost_window = recent[CRUSHED_GOOD_FRAMES:]

        good_avg = float(np.mean([_avg_kp_conf(tf) for tf in good_window]))
        lost_avg = float(np.mean([_avg_kp_conf(tf) for tf in lost_window]))

        if good_avg < CRUSHED_CONF_GOOD:
            return None
        if lost_avg > CRUSHED_CONF_LOST:
            return None

        # Extra sanity check -- bbox area must have shrunk significantly.
        # A worker walking BEHIND an occluder keeps roughly the same visible
        # bbox area; a worker who is being CRUSHED / TRAPPED typically has
        # only a portion of their body visible, so the bbox shrinks.
        good_areas = [
            float(max(0.0, (tf.bbox[2] - tf.bbox[0]) * (tf.bbox[3] - tf.bbox[1])))
            for tf in good_window
        ]
        lost_areas = [
            float(max(0.0, (tf.bbox[2] - tf.bbox[0]) * (tf.bbox[3] - tf.bbox[1])))
            for tf in lost_window
        ]
        good_area = float(np.mean(good_areas)) if good_areas else 0.0
        lost_area = float(np.mean(lost_areas)) if lost_areas else 0.0
        if good_area <= 0.0:
            return None
        shrink_ratio = lost_area / good_area
        if shrink_ratio > CRUSHED_BBOX_SHRINK_RATIO:
            # Bbox didn't shrink enough -- this looks like ordinary occlusion,
            # not a crush event.
            return None

        confidence = float(np.clip(0.5 + (good_avg - lost_avg), 0.5, 0.99))
        return AccidentEvent(
            frame      = frame_idx,
            track_id   = track.track_id,
            type       = "CRUSHED",
            severity   = "CRITICAL",
            confidence = confidence,
            reason     = (
                f"Tracked at avg_conf={good_avg:.2f} for {CRUSHED_GOOD_FRAMES} "
                f"frames, then collapsed to avg_conf={lost_avg:.2f} for "
                f"{CRUSHED_LOST_FRAMES} frames AND bbox shrank to "
                f"{shrink_ratio*100:.0f}% of baseline -- worker likely trapped or covered"
            ),
        )

    def _check_motionless_down(
        self, track: TrackedPerson, frame_idx: int, state: _TrackState,
    ) -> Optional[AccidentEvent]:
        # Only fires after a fall has been observed for this track.
        if state.last_fall_frame < 0:
            return None
        if self._in_cooldown(state, "MOTIONLESS_DOWN", frame_idx):
            return None

        elapsed_since_fall = frame_idx - state.last_fall_frame
        if elapsed_since_fall < self.motionless_down_frames:
            return None

        window = self._tail_window(track, self.motionless_down_frames)
        if len(window) < self.motionless_down_frames // 2:
            return None

        # All-horizontal check
        for tf in window:
            axis = body_axis_angle_from_horizontal(tf.kps_xy, tf.kps_conf)
            if axis is None:
                continue
            if axis > self.fallen_deg + 15.0:
                return None  # at any point in the window the person stood up

        # Low-motion check
        coms = self._com_series_window(window)
        if len(coms) < 4:
            return None
        vels = self._normalized_velocities(coms)
        if not vels:
            return None
        max_v = float(max(vels))
        if max_v > MOTIONLESS_VEL_NORM_MAX:
            return None

        return AccidentEvent(
            frame      = frame_idx,
            track_id   = track.track_id,
            type       = "MOTIONLESS_DOWN",
            severity   = "CRITICAL",
            confidence = 0.95,
            reason     = (
                f"Worker has remained horizontal and motionless for "
                f"{elapsed_since_fall} processed frames since fall "
                f"(max COM velocity {max_v:.4f} normalized)"
            ),
        )

    def _check_stumble(
        self, track: TrackedPerson, frame_idx: int, state: _TrackState,
    ) -> Optional[AccidentEvent]:
        if self._in_cooldown(state, "STUMBLE", frame_idx):
            return None
        # Suppress STUMBLE during/right after a real FALL — the dropping hip
        # is part of the same fall, not a separate stumble event.
        if state.last_fall_frame >= 0 and (
            (frame_idx - state.last_fall_frame) < EVENT_COOLDOWN_FRAMES["FALL"]
        ):
            return None

        window = self._tail_window(track, STUMBLE_RECOVERY_FRAMES + 2)
        if len(window) < 4:
            return None

        # Track hip-Y trajectory through window, normalized to frame height.
        ys: List[float] = []
        for tf in window:
            hip = hip_midpoint(tf.kps_xy, tf.kps_conf)
            if hip is None:
                ys.append(float("nan"))
            else:
                ys.append(float(hip[1]) / float(self.frame_h))
        ys_arr = np.asarray(ys, dtype=np.float32)
        if np.isnan(ys_arr).all():
            return None

        # Look for the pattern: starts low (small y), peaks high (large y),
        # returns to low. In image coords, larger y = lower in frame.
        # Use the half-window split.
        valid = ~np.isnan(ys_arr)
        if valid.sum() < 4:
            return None

        first_half  = ys_arr[: len(ys_arr) // 2][valid[: len(ys_arr) // 2]]
        second_half = ys_arr[len(ys_arr) // 2 :][valid[len(ys_arr) // 2 :]]
        if len(first_half) < 2 or len(second_half) < 2:
            return None

        # Largest drop in y (i.e. moved down in frame) within window
        drop = float(ys_arr[valid].max() - first_half.min())
        # And recovered: end y close to start y
        recovered = abs(float(second_half[-1] - first_half[0])) < 0.5 * drop

        if drop < STUMBLE_HIP_DROP_NORM or not recovered:
            return None

        # Person never went fully horizontal -- otherwise it would've been a FALL.
        latest_axis = body_axis_angle_from_horizontal(
            track.history[-1].kps_xy, track.history[-1].kps_conf,
        )
        if latest_axis is not None and latest_axis < 45.0:
            return None

        return AccidentEvent(
            frame      = frame_idx,
            track_id   = track.track_id,
            type       = "STUMBLE",
            severity   = "WARN",
            confidence = float(np.clip(0.4 + drop, 0.4, 0.85)),
            reason     = f"Hip dropped {drop * 100:.0f}% of frame height then recovered",
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _tail_window(self, track: TrackedPerson, n: int) -> List[TrackFrame]:
        if not track.history:
            return []
        return list(track.history)[-n:]

    def _com_series(self, track: TrackedPerson) -> List[Optional[np.ndarray]]:
        return [hip_midpoint(tf.kps_xy, tf.kps_conf) for tf in track.history]

    def _com_series_window(self, window: List[TrackFrame]) -> List[Optional[np.ndarray]]:
        return [hip_midpoint(tf.kps_xy, tf.kps_conf) for tf in window]

    def _normalized_velocities(self, coms: List[Optional[np.ndarray]]) -> List[float]:
        """
        Pairwise pixel velocities between consecutive non-None COMs,
        normalized by the frame diagonal so they're resolution-independent.

        Pairs across None gaps are skipped so a brief miss doesn't manifest
        as a fake huge velocity.
        """
        out: List[float] = []
        prev: Optional[np.ndarray] = None
        for cur in coms:
            if cur is None:
                prev = None
                continue
            if prev is not None:
                dx = float(cur[0] - prev[0])
                dy = float(cur[1] - prev[1])
                v_pix = math.hypot(dx, dy)
                out.append(v_pix / self.frame_diag)
            prev = cur
        return out

    def _in_cooldown(self, state: _TrackState, event_type: str, frame_idx: int) -> bool:
        last = state.last_event_frame.get(event_type, -10**9)
        return (frame_idx - last) < EVENT_COOLDOWN_FRAMES.get(event_type, 0)

    def _stamp_cooldown(self, state: _TrackState, event: AccidentEvent) -> None:
        state.last_event_frame[event.type] = event.frame


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _avg_kp_conf(tf: TrackFrame) -> float:
    if tf.kps_conf is None or tf.kps_conf.size == 0:
        return 0.0
    return float(np.mean(tf.kps_conf))


def overall_status(events: List[AccidentEvent]) -> str:
    """
    Reduce a list of events into a single status:

        CRITICAL  any CRITICAL event present
        WARN      only WARN events
        SAFE      no events
    """
    if not events:
        return "SAFE"
    if any(e.severity == "CRITICAL" for e in events):
        return "CRITICAL"
    return "WARN"


# ---------------------------------------------------------------------------
# Overlay renderer — draws active accident events on the output video
# ---------------------------------------------------------------------------

# BGR colours per severity. CRITICAL = saturated red, WARN = saturated orange.
_OVERLAY_STYLE: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # severity:    (border/banner BGR,    bbox highlight BGR)
    "CRITICAL":    ((30,  30, 230),       (40,  40, 240)),
    "WARN":        ((40, 130, 240),       (50, 140, 245)),
}
_OVERLAY_TEXT_BGR: Tuple[int, int, int] = (255, 255, 255)


# Verbose banner label, used when only one event is active — narrow portrait
# videos can't fit the full sentence for two events.
_EVENT_LABELS: Dict[str, str] = {
    "FALL":             "WORKER FALL",
    "STRUCK":           "POSSIBLE IMPACT",
    "CRUSHED":          "WORKER TRAPPED / OBSCURED",
    "MOTIONLESS_DOWN":  "WORKER DOWN / NOT MOVING",
    "STUMBLE":          "STUMBLE",
}

# Compact banner label, used when multiple events are active simultaneously.
_EVENT_LABELS_SHORT: Dict[str, str] = {
    "FALL":             "FALL",
    "STRUCK":           "STRUCK",
    "CRUSHED":          "TRAPPED",
    "MOTIONLESS_DOWN":  "DOWN",
    "STUMBLE":          "STUMBLE",
}


class AccidentOverlayRenderer:
    """
    Stateful overlay renderer that keeps recent accident events visible on
    the output video for ``ttl_frames`` after they fire, then fades out.

    Visual scheme
    -------------
        CRITICAL — pulsing thick red border + red top banner
        WARN     — solid orange border + orange top banner
    Per-track bounding boxes are highlighted in the same colour while their
    event is active. If the same (track_id, type) fires again before the TTL
    expires, the existing entry's TTL is refreshed instead of stacking.

    Usage
    -----
        renderer = AccidentOverlayRenderer()

        # In the per-frame loop:
        for ev in events_fired_this_frame:
            renderer.push(ev)

        # Right before writing the frame to disk:
        renderer.draw(frame, frame_idx, active_tracks)

    The renderer is independent from the detector so the detection pass and
    the rendering pass can be reasoned about separately.
    """

    def __init__(
        self,
        ttl_frames:           int = 45,    # ~3 s @ 15 fps effective
        border_thickness:     int = 12,
        banner_height_px:     int = 64,
    ) -> None:
        self._ttl              = int(ttl_frames)
        self._border_thickness = int(border_thickness)
        self._banner_h         = int(banner_height_px)
        self._active: List[Dict] = []   # entries: {"event": AccidentEvent, "remaining": int}

    # ------------------------------------------------------------------

    def push(self, event: AccidentEvent) -> None:
        """
        Register a new event. If the same (track_id, type) is already active,
        its TTL is refreshed and the event details (reason, conf) are
        replaced with the most recent observation.
        """
        for entry in self._active:
            existing = entry["event"]
            if existing.track_id == event.track_id and existing.type == event.type:
                entry["remaining"] = self._ttl
                entry["event"]     = event
                return
        self._active.append({"event": event, "remaining": self._ttl})

    def has_active_events(self) -> bool:
        return bool(self._active)

    # ------------------------------------------------------------------

    def draw(
        self,
        frame:    np.ndarray,
        frame_idx: int,
        tracks:   Optional[List[TrackedPerson]] = None,
    ) -> None:
        """
        Render all currently-active overlays on the frame in place, then
        decay TTLs and drop expired entries.

        Safe to call when there are no active events — a no-op then.
        """
        if not self._active:
            return

        h, w = frame.shape[:2]

        # Severity rank: CRITICAL renders on top of WARN.
        sev_rank = {"CRITICAL": 0, "WARN": 1}
        self._active.sort(key=lambda x: sev_rank.get(x["event"].severity, 99))
        worst_sev = self._active[0]["event"].severity

        border_color, _ = _OVERLAY_STYLE.get(worst_sev, _OVERLAY_STYLE["WARN"])

        # ---- Pulsing border (CRITICAL only) or steady (WARN) -----------
        if worst_sev == "CRITICAL":
            # Pulse thickness every 4 frames so it draws attention.
            pulse_thick = self._border_thickness + 4 if (frame_idx // 4) % 2 == 0 else self._border_thickness
        else:
            pulse_thick = self._border_thickness

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, pulse_thick)

        # ---- Top banner with all active events -------------------------
        banner_color, _ = _OVERLAY_STYLE.get(worst_sev, _OVERLAY_STYLE["WARN"])
        cv2.rectangle(frame, (0, 0), (w, self._banner_h), banner_color, -1)
        cv2.line(frame, (0, self._banner_h), (w, self._banner_h),
                 _OVERLAY_TEXT_BGR, 1, cv2.LINE_AA)

        # Banner text — concat active events, "FALL Worker#1 | STRUCK Worker#2"
        scale = max(1.0, h / 720.0)   # scale text with frame height
        font  = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9 * scale
        thickness  = max(2, int(round(2 * scale)))

        # Use verbose labels for a single event, compact for multi-event so
        # we can fit several into a narrow portrait-video banner. Per-track
        # numbering is intentionally omitted -- the affected workers are
        # already highlighted by their bbox outline + tag, so the banner
        # only needs to name what kind of event(s) are happening.
        seen_types: set = set()
        labels: List[str] = []
        if len(self._active) == 1:
            ev = self._active[0]["event"]
            labels.append(_EVENT_LABELS.get(ev.type, ev.type))
        else:
            for entry in self._active:
                ev = entry["event"]
                if ev.type in seen_types:
                    continue   # collapse duplicates (same event type, different workers)
                seen_types.add(ev.type)
                labels.append(_EVENT_LABELS_SHORT.get(ev.type, ev.type))
        banner_text = "  |  ".join(labels)

        # If still too wide, drop trailing events and tag with "+N more".
        max_text_width = w - int(20 * scale)
        if cv2.getTextSize(banner_text, font, font_scale, thickness)[0][0] > max_text_width:
            shown: List[str] = []
            for i, lab in enumerate(labels):
                trial_parts = shown + [lab]
                if i + 1 < len(labels):
                    trial = "  |  ".join(trial_parts) + f"  +{len(labels) - i - 1}"
                else:
                    trial = "  |  ".join(trial_parts)
                if cv2.getTextSize(trial, font, font_scale, thickness)[0][0] > max_text_width:
                    break
                shown.append(lab)
            if not shown:
                shown = [labels[0]]   # always show at least one even if it overflows
            extra = len(labels) - len(shown)
            banner_text = "  |  ".join(shown) + (f"  +{extra}" if extra > 0 else "")
        # Vertical center within banner
        (_, text_h), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)
        text_y = (self._banner_h + text_h) // 2 - baseline // 2
        cv2.putText(frame, banner_text, (int(10 * scale), text_y),
                    font, font_scale, _OVERLAY_TEXT_BGR, thickness, cv2.LINE_AA)

        # ---- Per-track bbox highlight ---------------------------------
        track_by_id: Dict[int, TrackedPerson] = {}
        if tracks is not None:
            for t in tracks:
                track_by_id[t.track_id] = t

        for entry in self._active:
            ev = entry["event"]
            t  = track_by_id.get(ev.track_id)
            if t is None or t.bbox is None:
                continue
            box = t.bbox
            x1 = int(max(0, min(w - 1, box[0])))
            y1 = int(max(0, min(h - 1, box[1])))
            x2 = int(max(0, min(w - 1, box[2])))
            y2 = int(max(0, min(h - 1, box[3])))
            if x2 <= x1 or y2 <= y1:
                continue

            _, hl_color = _OVERLAY_STYLE.get(ev.severity, _OVERLAY_STYLE["WARN"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), hl_color, max(3, int(round(4 * scale))))

            # Per-bbox event tag — verbose form, but if it would overflow
            # the frame on the right, clamp the x origin so it stays visible.
            tag = f"{_EVENT_LABELS.get(ev.type, ev.type)}  conf={ev.confidence:.2f}"
            tag_scale = 0.55 * scale
            tag_thick = max(1, int(round(1 * scale)))
            (tw, th), _ = cv2.getTextSize(tag, font, tag_scale, tag_thick)
            pad = int(4 * scale)
            tag_y0 = max(self._banner_h + 6, y1 - th - 2 * pad)
            tag_x0 = x1
            if tag_x0 + tw + 2 * pad > w:
                tag_x0 = max(0, w - (tw + 2 * pad) - 2)
            cv2.rectangle(frame, (tag_x0, tag_y0),
                          (tag_x0 + tw + 2 * pad, tag_y0 + th + 2 * pad),
                          hl_color, -1)
            cv2.putText(frame, tag, (tag_x0 + pad, tag_y0 + th + pad // 2),
                        font, tag_scale, _OVERLAY_TEXT_BGR, tag_thick, cv2.LINE_AA)

        # ---- Decay TTLs and remove expired ----------------------------
        for entry in self._active:
            entry["remaining"] -= 1
        self._active = [e for e in self._active if e["remaining"] > 0]
