"""
video_utils.py
==============
Shared video I/O helpers used across detection modules.
"""

import os
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def open_video(path: str) -> cv2.VideoCapture:
    """
    Open a video file and validate that it is readable.

    Parameters
    ----------
    path : str
        Absolute or relative path to the video file.

    Returns
    -------
    cv2.VideoCapture
        An opened capture object.

    Raises
    ------
    ValueError
        If the file cannot be opened or has zero frames.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open the video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video appears to be empty (0 frames): {path}")

    logger.info(
        "Opened video '%s' — %d frames @ %.1f FPS (%dx%d)",
        path,
        total_frames,
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
    """
    Create a VideoWriter for saving annotated output.

    Parameters
    ----------
    output_path : str
        Path where the output video will be written.
    fps : float
        Frames per second for the output video.
    width, height : int
        Frame dimensions.
    fourcc : str
        FourCC codec string (default ``"mp4v"``).

    Returns
    -------
    cv2.VideoWriter
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    codec  = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {output_path}")
    return writer


def cleanup_file(path: str) -> None:
    """
    Silently delete a file if it exists.

    Parameters
    ----------
    path : str
        Path to the file to be removed.
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Deleted temporary file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete file '%s': %s", path, exc)
