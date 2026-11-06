"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Cuts the final accepted segments out of the source video as .mp4 clips
(this is what the user / web demo actually watches).

- Preferred: ffmpeg (re-encodes so cuts are frame-accurate, keeps audio).
- Fallback : OpenCV VideoWriter if ffmpeg is not on PATH (video only, no audio).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional


def _ffmpeg_binary() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:  # optional: pip install imageio-ffmpeg ships a static binary
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_video_duration(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(n / fps) if fps > 0 else 0.0


def export_clip(video_path: str, start: float, end: float, out_path: str) -> str:
    """Write [start, end] of video_path to out_path. Returns the method used."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    start, end = max(0.0, float(start)), float(end)
    if end <= start:
        raise ValueError(f"Invalid clip range {start}-{end}")

    ffmpeg = _ffmpeg_binary()
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{end - start:.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-movflags", "+faststart", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return "ffmpeg"
        print(f"[Track3/export] ffmpeg failed ({proc.stderr.strip()[:200]}), falling back to OpenCV")

    _export_with_opencv(video_path, start, end, out_path)
    return "opencv"


def _export_with_opencv(video_path: str, start: float, end: float, out_path: str) -> None:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    n_frames = int(round((end - start) * fps))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
