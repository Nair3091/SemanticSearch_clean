import cv2
import numpy as np
import pytest

from src.common.types import Frame
from src.vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows)


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.avi")   # 10s @ 10fps, MJPG
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    w.release()
    return path


def test_coarse_frame_count_and_shape(video):
    frames = coarse_sample(video, 2.0)
    assert [f.timestamp for f in frames] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert frames[0].image.shape == (48, 64, 3)


def test_dense_frame_count(video):
    frames = dense_resample(video, 1.0, 2.0, 0.25)
    assert len(frames) == 5


def test_dense_clamped_to_duration(video):
    frames = dense_resample(video, 9.0, 50.0, 0.5)
    assert all(f.timestamp < 10.0 for f in frames)


def test_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        coarse_sample("nope.mp4", 1.0)


def _frames(n, step):
    return [Frame(timestamp=i * step, frame_index=i) for i in range(n)]


def test_window_merging():
    frames = _frames(20, 2.0)
    scores = np.zeros(20)
    scores[5], scores[6], scores[15] = 0.9, 0.7, 0.85
    assert select_promising_windows(frames, scores, 3, 3.0) == [(7.0, 15.0), (27.0, 33.0)]


def test_window_edge_cases():
    assert select_promising_windows([], np.array([]), 2, 3.0) == []
    with pytest.raises(ValueError):
        select_promising_windows(_frames(3, 1.0), np.zeros(2), 1, 1.0)