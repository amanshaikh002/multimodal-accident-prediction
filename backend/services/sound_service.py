"""
sound_service.py
================
Anomaly sound detection service.

Pipeline (per video)
--------------------
1. Extract mono audio at 22 050 Hz from the input video. Tries librosa's
   built-in ffmpeg-backed loader first; falls back to a direct ffmpeg
   subprocess if that fails (e.g. older librosa without av support).
2. Slide a 3-second window over the audio with a 1-second hop, so we get
   one prediction per second of footage.
3. For each window, compute the 13-band MFCC mean (matches the training
   feature pipeline in ``Sound Anomaly/preprocess (1).py``).
4. Run the cached RandomForest classifier (``audio_model.pkl``):
       0 = NORMAL,  1 = ANOMALY
5. Apply temporal persistence: a window only counts as anomalous if the
   model fires on _PERSISTENCE_WINDOWS in a row. Suppresses single-window
   flicker from background bumps.
6. Group confirmed anomalous windows into contiguous *events* with
   start_sec / end_sec / max_confidence / avg_confidence.
7. Aggregate:
       anomaly_ratio = anomaly_windows / total_windows
       status = UNSAFE if ratio > _ANOMALY_RATIO_THRESHOLD
                 OR any single event >= _LONG_EVENT_SEC
8. If ``output_video_path`` is given, copy the input video there so the
   frontend has something to play with audio while reading the analysis.

Returns
-------
{
    "module":           "sound",
    "status":           "SAFE" | "UNSAFE",
    "anomaly_detected": bool,
    "anomaly_ratio":    float,
    "total_windows":    int,
    "anomaly_windows":  int,
    "duration_sec":     float,
    "events": [
        {
            "start_sec":      float,
            "end_sec":        float,
            "duration_sec":   float,
            "avg_confidence": float,
            "max_confidence": float,
        }, ...
    ],
    "message":      str,
    "output_video": str | None,
}
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent.parent              # backend/
_MODEL_PATH = str(_HERE / "Sound Anomaly" / "audio_model.pkl")
_OUTPUT_DIR = _HERE / "temp" / "output"

_TARGET_SR        = 22_050      # Hz; matches librosa's default
_N_MFCC           = 13          # must match training feature pipeline
_WINDOW_SEC       = 3.0         # must match training duration=3
_HOP_SEC          = 1.0         # one prediction per second of audio

# Number of consecutive anomalous windows required before counting as a
# real anomaly. 2 = ~1 sec of sustained signal (with 1-sec hop).
_PERSISTENCE_WINDOWS = 2

# Verdict thresholds
_ANOMALY_RATIO_THRESHOLD = 0.05      # > 5% of windows -> UNSAFE
_LONG_EVENT_SEC          = 2.0       # any single confirmed event >= 2s -> UNSAFE


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[Any] = None


def _get_model() -> Any:
    """Return the cached audio classifier, loading it on first call."""
    global _model
    if _model is None:
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"Audio model not found at '{_MODEL_PATH}'. "
                "Place audio_model.pkl in backend/Sound Anomaly/."
            )
        logger.info("[SOUND] Loading audio model: %s", _MODEL_PATH)
        _model = joblib.load(_MODEL_PATH)
        n_in = getattr(_model, "n_features_in_", "?")
        classes = getattr(_model, "classes_", "?")
        logger.info("[SOUND] Model loaded -- n_features_in=%s  classes=%s",
                    n_in, classes)
    return _model


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> Optional[str]:
    """
    Return a path to a usable ffmpeg binary.

    Priority:
      1. ``imageio_ffmpeg.get_ffmpeg_exe()`` — bundled with our existing
         ``imageio-ffmpeg`` dep, always available on Windows / macOS / Linux
         regardless of system PATH.
      2. The bare string ``"ffmpeg"`` (relies on system PATH).
      3. None — caller will know to give up.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        path = get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return "ffmpeg"   # fall back to PATH lookup; subprocess will tell us if missing


def _extract_audio(video_path: str, sr: int = _TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load mono audio waveform from a video file.

    Strategy:
      1. Ask librosa to load the video directly (uses soundfile / audioread
         under the hood; works for most mp4 builds with ffmpeg on PATH).
      2. If librosa returns nothing or raises, transcode with the bundled
         ffmpeg into a temp wav and re-load that wav.

    Returns (waveform, sample_rate).
    Raises ValueError when the video genuinely has no audio (or the audio
    track can't be decoded) — caller is expected to convert that into a
    friendly "no audio analysis possible" response, NOT a hard error.
    """
    import librosa

    # ---- Attempt 1: librosa direct load ----------------------------------
    try:
        waveform, sr_out = librosa.load(video_path, sr=sr, mono=True)
        if waveform.size > 0:
            return waveform.astype(np.float32), int(sr_out)
        logger.info("[SOUND] librosa returned empty waveform -- falling back to ffmpeg.")
    except Exception as exc:
        logger.info(
            "[SOUND] librosa direct load failed (%s: %s) -- falling back to ffmpeg.",
            type(exc).__name__, exc,
        )

    # ---- Attempt 2: transcode with bundled ffmpeg, then load wav --------
    ffmpeg_bin = _resolve_ffmpeg()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(sr),
            wav_path,
        ]
        logger.info("[SOUND] ffmpeg transcoding via: %s", ffmpeg_bin)
        try:
            proc = subprocess.run(
                cmd,
                check=False,                # we'll inspect ourselves
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise ValueError(
                "ffmpeg binary not available -- cannot extract audio. "
                "Reinstall imageio-ffmpeg or add ffmpeg to PATH."
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip().splitlines()
            tail = " | ".join(stderr[-3:]) if stderr else "no stderr"
            logger.warning(
                "[SOUND] ffmpeg exited %d during audio extraction: %s",
                proc.returncode, tail,
            )
            raise ValueError(
                f"Video has no extractable audio track "
                f"(ffmpeg exit {proc.returncode}: {tail[:200]})"
            )

        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            raise ValueError("Video has no audio track.")

        waveform, sr_out = librosa.load(wav_path, sr=sr, mono=True)
        if waveform.size == 0:
            raise ValueError("Extracted audio is empty.")
        return waveform.astype(np.float32), int(sr_out)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Feature extraction (matches training)
# ---------------------------------------------------------------------------

def _mfcc_mean(window: np.ndarray, sr: int, n_mfcc: int = _N_MFCC) -> np.ndarray:
    """13-dim mean across MFCC bands -- exact same shape the model was trained on."""
    import librosa
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding window prediction
# ---------------------------------------------------------------------------

def _slide_predict(
    model:    Any,
    waveform: np.ndarray,
    sr:       int,
) -> List[Dict[str, Any]]:
    """
    Slide a fixed window over the waveform; return per-window predictions.

    Returns a list of dicts (oldest first):
        {start_sec, end_sec, pred (0/1), confidence (max prob), prob_anomaly}
    """
    n_samples = len(waveform)
    win  = int(_WINDOW_SEC * sr)
    hop  = int(_HOP_SEC    * sr)

    # If the audio is shorter than one window, pad with zeros and run once.
    if n_samples < win:
        padded = np.zeros(win, dtype=waveform.dtype)
        padded[:n_samples] = waveform
        feat = _mfcc_mean(padded, sr).reshape(1, -1)
        pred = int(model.predict(feat)[0])
        prob = model.predict_proba(feat)[0]
        return [{
            "start_sec":     0.0,
            "end_sec":       round(n_samples / sr, 3),
            "pred":          pred,
            "confidence":    float(np.max(prob)),
            "prob_anomaly":  float(prob[1]) if len(prob) > 1 else 0.0,
        }]

    out: List[Dict[str, Any]] = []
    for start in range(0, n_samples - win + 1, hop):
        chunk  = waveform[start:start + win]
        feat   = _mfcc_mean(chunk, sr).reshape(1, -1)
        pred   = int(model.predict(feat)[0])
        prob   = model.predict_proba(feat)[0]
        out.append({
            "start_sec":     round(start / sr, 3),
            "end_sec":       round((start + win) / sr, 3),
            "pred":          pred,
            "confidence":    float(np.max(prob)),
            "prob_anomaly":  float(prob[1]) if len(prob) > 1 else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Persistence + event grouping
# ---------------------------------------------------------------------------

def _confirm_with_persistence(
    raw_predictions: List[Dict[str, Any]],
    persistence:     int = _PERSISTENCE_WINDOWS,
) -> List[bool]:
    """
    Mark each window as a *confirmed* anomaly only when it is part of a
    run of >= ``persistence`` consecutive raw-anomaly windows. Returns a
    list of booleans aligned with raw_predictions.
    """
    confirmed = [False] * len(raw_predictions)
    streak    = 0
    for i, p in enumerate(raw_predictions):
        if p["pred"] == 1:
            streak += 1
            if streak >= persistence:
                # Backfill any earlier windows in this run that hadn't yet
                # met the threshold so events have crisp start times.
                for j in range(max(0, i - streak + 1), i + 1):
                    confirmed[j] = True
        else:
            streak = 0
    return confirmed


def _group_into_events(
    predictions: List[Dict[str, Any]],
    confirmed:   List[bool],
) -> List[Dict[str, Any]]:
    """Collapse contiguous confirmed-anomaly windows into single events."""
    events: List[Dict[str, Any]] = []
    cur:   Optional[Dict[str, Any]] = None

    for p, ok in zip(predictions, confirmed):
        if ok:
            if cur is None:
                cur = {
                    "start_sec":     p["start_sec"],
                    "end_sec":       p["end_sec"],
                    "confidences":   [p["confidence"]],
                    "anomaly_probs": [p["prob_anomaly"]],
                }
            else:
                cur["end_sec"]       = p["end_sec"]
                cur["confidences"].append(p["confidence"])
                cur["anomaly_probs"].append(p["prob_anomaly"])
        else:
            if cur is not None:
                events.append(_finalize_event(cur))
                cur = None
    if cur is not None:
        events.append(_finalize_event(cur))
    return events


def _finalize_event(cur: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_sec":      round(float(cur["start_sec"]), 3),
        "end_sec":        round(float(cur["end_sec"]),   3),
        "duration_sec":   round(float(cur["end_sec"] - cur["start_sec"]), 3),
        "avg_confidence": round(float(np.mean(cur["confidences"])),       3),
        "max_confidence": round(float(np.max (cur["confidences"])),       3),
        "max_anomaly_prob": round(float(np.max(cur["anomaly_probs"])),    3),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_sound_video(
    video_path:        str,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run anomaly sound detection over the audio track of a video file."""
    model = _get_model()

    # ---- 1. Extract audio --------------------------------------------------
    # ANY extraction failure (no audio track, codec issue, ffmpeg missing,
    # corrupt file...) becomes a graceful SAFE result with an informative
    # message. We never return 4xx for "no audio" -- the request was valid,
    # the audio just wasn't analysable.
    try:
        waveform, sr = _extract_audio(video_path, sr=_TARGET_SR)
    except Exception as exc:
        logger.info("[SOUND] Audio extraction failed: %s", exc)
        return _empty_result(
            video_path        = video_path,
            output_video_path = output_video_path,
            message = (
                "No audio could be extracted from this video "
                f"({type(exc).__name__}: {str(exc)[:160]}). "
                "Sound anomaly analysis was skipped; the rest of the pipeline is unaffected."
            ),
        )

    duration_sec = round(len(waveform) / sr, 2)
    logger.info("[SOUND] Audio loaded: %.2fs @ %d Hz (%d samples)",
                duration_sec, sr, len(waveform))

    # ---- 2. Sliding window predict ----------------------------------------
    # Wrap in a safety net so a librosa / sklearn edge case (e.g. a NaN
    # frame from an unusual codec) downgrades to a friendly SAFE result
    # instead of bubbling up as a 5xx.
    try:
        raw_predictions = _slide_predict(model, waveform, sr)
    except Exception as exc:
        logger.warning("[SOUND] Sliding-window prediction failed: %s", exc)
        return _empty_result(
            video_path        = video_path,
            output_video_path = output_video_path,
            message = (
                f"Audio analysis aborted ({type(exc).__name__}: {str(exc)[:160]}). "
                "The waveform was extracted but feature/prediction failed."
            ),
        )
    total_windows   = len(raw_predictions)

    # ---- 3. Persistence + events ------------------------------------------
    confirmed       = _confirm_with_persistence(raw_predictions)
    events          = _group_into_events(raw_predictions, confirmed)
    anomaly_windows = sum(1 for c in confirmed if c)

    # ---- 4. Aggregate verdict ---------------------------------------------
    anomaly_ratio = (anomaly_windows / total_windows) if total_windows > 0 else 0.0
    long_event    = any(e["duration_sec"] >= _LONG_EVENT_SEC for e in events)
    is_unsafe     = (anomaly_ratio > _ANOMALY_RATIO_THRESHOLD) or long_event
    status        = "UNSAFE" if is_unsafe else "SAFE"

    if is_unsafe:
        worst = max(events, key=lambda e: e["max_anomaly_prob"], default=None)
        if worst is not None:
            message = (
                f"Anomalous sound detected from {worst['start_sec']:.1f}s to "
                f"{worst['end_sec']:.1f}s (max anomaly prob {worst['max_anomaly_prob']:.2f}). "
                f"Total {len(events)} event(s) across {anomaly_windows} window(s)."
            )
        else:
            message = "Anomalous sound activity detected."
    else:
        message = (
            f"No anomalous sounds detected across {total_windows} sampled window(s) "
            f"({duration_sec:.1f}s of audio)."
        )

    # ---- 5. Optional video copy (sound has no natural visual annotation) --
    out_filename = _maybe_copy_video(video_path, output_video_path)

    result = {
        "module":           "sound",
        "status":           status,
        "anomaly_detected": is_unsafe,
        "anomaly_ratio":    round(anomaly_ratio, 4),
        "total_windows":    int(total_windows),
        "anomaly_windows":  int(anomaly_windows),
        "duration_sec":     duration_sec,
        "events":           events,
        "message":          message,
        "output_video":     out_filename,
    }

    logger.info(
        "[SOUND] Done -- %d windows | %d anomalous (%.1f%%) | "
        "%d event(s) | longest=%.1fs | status=%s",
        total_windows, anomaly_windows, anomaly_ratio * 100,
        len(events),
        max((e["duration_sec"] for e in events), default=0.0),
        status,
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result(
    video_path:        str,
    output_video_path: Optional[str],
    message:           str,
) -> Dict[str, Any]:
    """Return a SAFE result for cases where no audio analysis was possible."""
    out_filename = _maybe_copy_video(video_path, output_video_path)
    return {
        "module":           "sound",
        "status":           "SAFE",
        "anomaly_detected": False,
        "anomaly_ratio":    0.0,
        "total_windows":    0,
        "anomaly_windows":  0,
        "duration_sec":     0.0,
        "events":           [],
        "message":          message,
        "output_video":     out_filename,
    }


def _maybe_copy_video(video_path: str, output_video_path: Optional[str]) -> Optional[str]:
    """
    Copy the input video to ``output_video_path`` so the frontend has
    something to play (sound mode has no natural visual annotation).
    Returns the basename used by the static /output mount, or None.
    """
    if not output_video_path:
        return None
    try:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video_path, output_video_path)
        return os.path.basename(output_video_path)
    except Exception as exc:
        logger.warning("[SOUND] Could not copy video to %s: %s", output_video_path, exc)
        return None
