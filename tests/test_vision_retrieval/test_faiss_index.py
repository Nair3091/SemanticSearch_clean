import numpy as np
import pytest
from src.vision_retrieval.indexing.faiss_index import build_index, load_index, search


def test_round_trip(tmp_path):
    v = np.random.rand(20, 512).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    path = str(tmp_path / "i.faiss")
    build_index(v, path)
    scores, idx = search(load_index(path), v[5], top_k=3)
    assert idx[0] == 5 and scores[0] > 0.999


def test_bad_shape_and_missing(tmp_path):
    with pytest.raises(ValueError):
        build_index(np.zeros(512, dtype=np.float32), str(tmp_path / "x.faiss"))
    with pytest.raises(FileNotFoundError):
        load_index(str(tmp_path / "missing.faiss"))