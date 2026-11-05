import numpy as np
from src.vision_retrieval.embedding.clip_encoder import score_frames


def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_score_shape():
    a, b = _norm(np.random.rand(5, 512)), _norm(np.random.rand(3, 512))
    assert score_frames(a, b).shape == (5, 3)


def test_score_empty():
    assert score_frames(np.zeros((0, 512)), np.zeros((2, 512))).shape == (0, 2)


def test_self_similarity_is_one():
    a = _norm(np.random.rand(4, 512))
    assert np.allclose(np.diag(score_frames(a, a)), 1.0, atol=1e-5)