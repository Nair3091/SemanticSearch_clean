from typing import List, Tuple
import cv2
import numpy as np

from src.common.types import Frame


def coarse_sample(video_path: str, interval_seconds: float) -> List[Frame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    frames: List[Frame] = []
    timestamp = 0.0
    while timestamp < duration:
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr_frame = cap.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frames.append(Frame(timestamp=timestamp, frame_index=frame_index, image=rgb_frame))
        timestamp += interval_seconds

    cap.release()
    return frames


def dense_resample(video_path: str, window_start: float, window_end: float,
                    interval_seconds: float) -> List[Frame]:

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    frames: List[Frame] = []
    timestamp = window_start
    while timestamp <= window_end and timestamp < duration:
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr_frame = cap.read()
        if not ok:
            break
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frames.append(Frame(timestamp=timestamp, frame_index=frame_index, image=rgb_frame))
        timestamp += interval_seconds

    cap.release()
    return frames


def select_promising_windows(
    frames: List[Frame],
    scores: np.ndarray,
    top_k: int,
    window_radius_seconds: float,
) -> List[Tuple[float, float]]:
    if len(frames) == 0:
        return []
    if len(frames) != len(scores):
        raise ValueError("frames and scores must be the same length")

    k = min(top_k, len(frames))
    # indices of the k highest scores, highest first
    top_indices = np.argsort(scores)[::-1][:k]

    raw_windows = []
    for idx in top_indices:
        center = frames[idx].timestamp
        start = max(0.0, center - window_radius_seconds)
        end = center + window_radius_seconds
        raw_windows.append((start, end))

    raw_windows.sort(key=lambda w: w[0])

    merged: List[Tuple[float, float]] = []
    for start, end in raw_windows:
        if merged and start <= merged[-1][1]:
            # overlaps the previous window — merge instead of duplicating work
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
