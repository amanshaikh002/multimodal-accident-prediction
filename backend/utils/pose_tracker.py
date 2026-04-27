"""
pose_tracker.py
================
Lightweight per-person tracker for the pose pipeline.

Why
---
The existing pose service re-selects a "primary person" every frame via
largest-bbox heuristic. That breaks temporal continuity: velocity and
acceleration buffers flip between workers when the pick changes mid-clip,
making any time-series safety logic (fall detection, sudden-impact spikes)
unreliable.

This tracker assigns a stable integer ID to each person across frames via
greedy IoU matching, and keeps a short rolling history per track so any
downstream module can ask "what has this specific person been doing?".

It is deliberately simple — no Kalman filter, no DeepSORT, no embeddings.
For workplace cameras (1-5 people, mostly stationary backgrounds), greedy
IoU matching at every processed frame is more than enough.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU of two xyxy boxes. Returns 0.0 for non-overlapping or zero-area."""
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Track entry & per-frame history record
# ---------------------------------------------------------------------------

@dataclass
class TrackFrame:
    """Snapshot of one tracked person at one processed frame."""
    frame_idx:  int
    bbox:       np.ndarray   # (4,) xyxy
    kps_xy:     np.ndarray   # (17, 2)
    kps_conf:   np.ndarray   # (17,)


@dataclass
class TrackedPerson:
    """A single tracked person across multiple frames."""
    track_id:        int
    history:         Deque[TrackFrame] = field(default_factory=lambda: deque(maxlen=60))
    last_seen_frame: int = -1
    age:             int = 0   # number of processed frames the track has existed
    misses:          int = 0   # consecutive frames without a detection match

    # ---- Convenience accessors (all return None if history is empty) ----

    @property
    def latest(self) -> Optional[TrackFrame]:
        return self.history[-1] if self.history else None

    @property
    def bbox(self) -> Optional[np.ndarray]:
        latest = self.latest
        return latest.bbox if latest is not None else None

    @property
    def kps_xy(self) -> Optional[np.ndarray]:
        latest = self.latest
        return latest.kps_xy if latest is not None else None

    @property
    def kps_conf(self) -> Optional[np.ndarray]:
        latest = self.latest
        return latest.kps_conf if latest is not None else None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PoseTracker:
    """
    Greedy IoU-based multi-person tracker.

    Usage
    -----
        tracker = PoseTracker()
        for frame_idx, (boxes, kps_xy_all, kps_conf_all) in enumerate(stream):
            active = tracker.update(frame_idx, boxes, kps_xy_all, kps_conf_all)
            for track in active:
                # use track.history, track.track_id, etc.

    Parameters
    ----------
    iou_thresh       : Minimum IoU for a detection-to-track match.
                       0.30 works well for handheld + workplace footage where
                       people drift between frames; raise to 0.50 if your
                       cameras are static and people move slowly.
    max_age_frames   : A track that has not been matched for this many
                       processed frames is dropped. Default 15 ≈ 1 sec at
                       stride 2 / 30 fps.
    history_len      : How many recent frames to keep per track. 60 ≈ 4 sec
                       at stride 2 / 30 fps — enough for fall detection
                       (1.5 s window) and post-fall motionless timer (3 s).
    """

    def __init__(
        self,
        iou_thresh:     float = 0.30,
        max_age_frames: int   = 15,
        history_len:    int   = 60,
    ) -> None:
        self.iou_thresh     = float(iou_thresh)
        self.max_age_frames = int(max_age_frames)
        self.history_len    = int(history_len)

        self._tracks:  Dict[int, TrackedPerson] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx:    int,
        boxes:        np.ndarray,
        kps_xy_all:   np.ndarray,
        kps_conf_all: np.ndarray,
    ) -> List[TrackedPerson]:
        """
        Match new detections to existing tracks; spawn new tracks as needed;
        retire stale tracks. Returns the list of currently active tracks.

        Parameters
        ----------
        frame_idx    : Monotonically increasing processed-frame index.
        boxes        : (N, 4) xyxy float array. May be empty.
        kps_xy_all   : (N, 17, 2) float array of keypoint coords.
        kps_conf_all : (N, 17)    float array of keypoint confidences.
        """
        n_dets = 0 if boxes is None else int(boxes.shape[0])

        if n_dets == 0:
            self._age_unmatched(frame_idx, matched_track_ids=set())
            return self._active_tracks()

        # --- Build IoU matrix between detections and existing tracks ----
        track_ids: List[int] = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]

        matches: List[Tuple[int, int]] = []          # (det_idx, track_idx_in_list)
        used_dets:   set = set()
        used_tracks: set = set()

        if track_ids:
            iou_matrix = np.zeros((n_dets, len(track_ids)), dtype=np.float32)
            for di in range(n_dets):
                for ti, tbox in enumerate(track_boxes):
                    if tbox is None:
                        continue
                    iou_matrix[di, ti] = _iou(boxes[di], tbox)

            # Greedy: pick highest-IoU pair above threshold, repeat.
            while True:
                best = float(self.iou_thresh)
                pick: Optional[Tuple[int, int]] = None
                for di in range(n_dets):
                    if di in used_dets:
                        continue
                    for ti in range(len(track_ids)):
                        if ti in used_tracks:
                            continue
                        v = float(iou_matrix[di, ti])
                        if v > best:
                            best = v
                            pick = (di, ti)
                if pick is None:
                    break
                matches.append(pick)
                used_dets.add(pick[0])
                used_tracks.add(pick[1])

        # --- Apply matches: append new frame data to existing tracks ----
        matched_track_ids: set = set()
        for di, ti in matches:
            track = self._tracks[track_ids[ti]]
            self._append_frame(track, frame_idx, boxes[di], kps_xy_all[di], kps_conf_all[di])
            matched_track_ids.add(track.track_id)

        # --- Spawn new tracks for unmatched detections -----------------
        for di in range(n_dets):
            if di in used_dets:
                continue
            new_track = TrackedPerson(
                track_id = self._next_id,
                history  = deque(maxlen=self.history_len),
            )
            self._next_id += 1
            self._append_frame(new_track, frame_idx, boxes[di], kps_xy_all[di], kps_conf_all[di])
            self._tracks[new_track.track_id] = new_track
            matched_track_ids.add(new_track.track_id)

        # --- Age unmatched tracks; retire those past max_age -----------
        self._age_unmatched(frame_idx, matched_track_ids)

        return self._active_tracks()

    def reset(self) -> None:
        """Clear all tracks. Call this between videos."""
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_frame(
        self,
        track:    TrackedPerson,
        frame_idx: int,
        bbox:     np.ndarray,
        kps_xy:   np.ndarray,
        kps_conf: np.ndarray,
    ) -> None:
        track.history.append(TrackFrame(
            frame_idx = int(frame_idx),
            bbox      = np.asarray(bbox,     dtype=np.float32).copy(),
            kps_xy    = np.asarray(kps_xy,   dtype=np.float32).copy(),
            kps_conf  = np.asarray(kps_conf, dtype=np.float32).copy(),
        ))
        track.last_seen_frame = int(frame_idx)
        track.age            += 1
        track.misses          = 0

    def _age_unmatched(self, frame_idx: int, matched_track_ids: set) -> None:
        """Increment miss count on unmatched tracks; drop those past max_age."""
        to_drop: List[int] = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids:
                continue
            track.misses += 1
            if track.misses > self.max_age_frames:
                to_drop.append(tid)
        for tid in to_drop:
            del self._tracks[tid]

    def _active_tracks(self) -> List[TrackedPerson]:
        """Tracks that received a detection on the most recent update."""
        return [t for t in self._tracks.values() if t.misses == 0]

    # Useful for debugging / logs
    @property
    def all_tracks(self) -> List[TrackedPerson]:
        return list(self._tracks.values())
