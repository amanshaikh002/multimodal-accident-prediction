"""
video_utils.py
==============
Shared video I/O helpers used by all detection modules.
"""

import logging
import os

import cv2

logger = logging.getLogger(__name__)


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
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """Create a VideoWriter for saving annotated output."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    codec  = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")
    return writer


def cleanup_file(path: str) -> None:
    """Silently delete a file if it exists."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Deleted temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete '%s': %s", path, exc)
