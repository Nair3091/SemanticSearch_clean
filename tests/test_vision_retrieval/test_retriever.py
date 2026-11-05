import cv2
import numpy as np
import pytest

from src.common.types import SubQuery
from src.vision_retrieval import retriever


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.avi")   # 10s @ 10fps, frame i has gray value 2*i (gets brighter)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    w.release()
    return path


@pytest.fixture(autouse=True)
def fake_clip(monkeypatch):
    def fake_embed_frames(frames):
        out = []
        for f in frames:
            p = float(f.image.mean()) / 255.0          # brightness in [0, 1]
            v = np.array([p, 1.0 - p], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out) if out else np.zeros((0, 2), dtype=np.float32)

    def fake_embed_text(texts):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    cfg = {"vision_retrieval": {"pass1_interval_seconds": 2.0, "pass2_interval_seconds": 0.25,
                                "top_k_windows": 1, "window_radius_seconds": 1.0}}
    monkeypatch.setattr(retriever, "embed_frames", fake_embed_frames)
    monkeypatch.setattr(retriever, "embed_text", fake_embed_text)
    monkeypatch.setattr(retriever, "load_config", lambda: cfg)


def test_window_follows_brightest_frame(video):
    r = retriever.retrieve(video, [SubQuery("bright", False)])
    assert r.windows == [(7.0, 9.0)]                 # brightest Pass-1 frame is t=8


def test_output_shape_and_dense_flags(video):
    r = retriever.retrieve(video, [SubQuery("bright", False)])
    # 5 coarse (0,2,4,6,8); t=8 is replaced by 9 dense frames (7.0..9.0) -> 4 + 9 = 13
    assert len(r.timestamps) == 13
    assert int(r.is_dense.sum()) == 9
    assert bool((r.timestamps[1:] > r.timestamps[:-1]).all())
    assert [s.shape for s in r.scores] == [(13,)]


def test_scores_align_with_sub_queries(video):
    r = retriever.retrieve(video, [SubQuery("a", False), SubQuery("b", True)])
    assert len(r.scores) == 2 and r.scores[0].shape == r.scores[1].shape


def test_negated_query_ignored_for_window_guidance(video):
    r = retriever.retrieve(video, [SubQuery("bright", False), SubQuery("bright", True)])
    assert r.windows == [(7.0, 9.0)]


def test_all_negated_picks_weakest_frames(video):
    r = retriever.retrieve(video, [SubQuery("bright", True)])
    assert r.windows == [(0.0, 1.0)]                 # darkest Pass-1 frame is t=0


def test_empty_sub_queries_raises(video):
    with pytest.raises(ValueError):
        retriever.retrieve(video, [])