"""
video_utils.py
==============
Shared video I/O helpers used by all detection modules.

Codec strategy for browser-compatible MP4
------------------------------------------
OpenCV on Windows (without Cisco OpenH264 DLL) cannot encode H.264.
Only `mp4v` / MPEG-4 Part 2 is available, which some browsers don't
support natively.

Strategy:
  1. Write annotated frames with OpenCV using `mp4v` to a temp file.
  2. After the writer is released, use imageio_ffmpeg (bundled binary)
     to re-encode to H.264 (libx264) in-place — guaranteed browser support.
  3. If imageio_ffmpeg is unavailable the mp4v file is served as-is
     (Chrome and Edge can still play it; Firefox/Safari may struggle).
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: H.264 re-encode via bundled FFmpeg
# ---------------------------------------------------------------------------

def _get_ffmpeg_exe() -> Optional[str]:
    """Return the path to a usable FFmpeg binary.

    Priority:
    1. imageio_ffmpeg bundled binary (no system install needed)
    2. System `ffmpeg` on PATH
    """
    # 1. Try imageio_ffmpeg first
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        logger.debug("Using imageio_ffmpeg binary: %s", exe)
        return exe
    except Exception as e:
        logger.debug("imageio_ffmpeg unavailable (%s) — trying system ffmpeg.", e)

    # 2. Fall back to system ffmpeg
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        logger.debug("Using system ffmpeg: %s", sys_ffmpeg)
        return sys_ffmpeg

    logger.warning(
        "No FFmpeg binary found (imageio_ffmpeg not installed AND 'ffmpeg' not on PATH). "
        "Run: pip install imageio-ffmpeg"
    )
    return None


def _ffmpeg_reencode(src: str, dst: str) -> bool:
    """
    Re-encode *src* → *dst* using H.264 / yuv420p for full browser
    compatibility. Uses the imageio_ffmpeg bundled binary so no system
    FFmpeg is required.

    Returns True on success.
    """
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg is None:
        logger.warning("No FFmpeg binary available — skipping H.264 re-encode.")
        return False

    tmp = dst + ".reenc.mp4"
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",     # mandatory for browser compat
        "-movflags", "+faststart",  # progressive streaming
        "-an",                      # no audio channel
        tmp,
    ]
    logger.info("[FFmpeg] Re-encoding: %s → %s", src, tmp)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            logger.warning("FFmpeg re-encode failed (code %d):\n%s", result.returncode, err[-2000:])
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

        # Sanity check: temp output must be non-empty
        if not os.path.exists(tmp) or os.path.getsize(tmp) < 1024:
            logger.warning("FFmpeg produced an empty/missing output file: %s", tmp)
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

        os.replace(tmp, dst)
        logger.info("H.264 re-encode complete → %s  (%.2f MB)",
                    dst, os.path.getsize(dst) / 1_048_576)
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg re-encode timed out.")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    except Exception as exc:
        logger.error("FFmpeg re-encode error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_video(path: str) -> cv2.VideoCapture:
    """
    Open and validate a video file.

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : OpenCV cannot open it, or it has 0 frames
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open the video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Video appears to be empty (0 frames): {path}")

    logger.info(
        "Opened video '%s' — %d frames @ %.1f FPS (%dx%d)",
        path, total,
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return cap


def get_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",    # mp4v is the most reliable on Windows OpenCV
) -> cv2.VideoWriter:
    """
    Create a VideoWriter for saving annotated output.

    Uses mp4v by default (works on all Windows OpenCV builds).
    Call finalize_video() after writer.release() to convert to H.264.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    codec  = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter for '{output_path}' with codec '{fourcc}'"
        )
    logger.info("VideoWriter ready — codec=%s  fps=%.1f  size=%dx%d  output=%s",
                fourcc, fps, width, height, output_path)
    return writer


def finalize_video(path: str) -> str:
    """
    Convert the OpenCV-written video to browser-compatible H.264 MP4.

    Call this immediately after writer.release().  The re-encode happens
    in-place (overwrites the original file).

    If imageio_ffmpeg is not available the file is left as-is.
    """
    if not os.path.isfile(path):
        logger.warning("finalize_video: file not found: %s", path)
        return path

    size_mb = os.path.getsize(path) / 1_048_576
    logger.info("Finalizing video '%s' (%.2f MB) → H.264 re-encode…", path, size_mb)

    if _ffmpeg_reencode(path, path):
        logger.info("Video finalized successfully as H.264 MP4.")
    else:
        logger.warning(
            "H.264 re-encode skipped — serving mp4v file directly. "
            "Chrome/Edge will play it; Firefox/Safari may not."
        )
    return path


def cleanup_file(path: str) -> None:
    """Silently delete a file if it exists."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Deleted temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete '%s': %s", path, exc)
