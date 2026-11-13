import numpy as np
import pytest

from src.common.types import Frame
from src.track2_vision_retrieval.embedding import frame_cache
from src.track2_vision_retrieval.scoring import softmax_confidence as sc


def kept(query):
    return {name for name, keep in zip(sc.COCO, sc.competitors(query)) if keep}


def test_bank_is_the_80_coco_objects():
    assert len(sc.COCO) == len(set(sc.COCO)) == 80 == len(sc.BANK)


def test_query_does_not_compete_with_itself_or_shared_words():
    assert "cat" not in kept("a cat")
    assert "bus" not in kept("a bus on a street") and "person" in kept("a bus on a street")
    assert "cup" not in kept("a cup of coffee")
    assert "car" not in kept("two red cars")                       # plural


def test_category_words_drop_their_members():
    assert not {"cat", "dog", "horse", "bird"} & kept("an animal")
    assert not {"bus", "car", "train"} & kept("a vehicle")
    assert not {"cat", "dog"} & kept("a pet") and "horse" in kept("a pet")
    assert "person" not in kept("a man walking")


def test_unrelated_query_keeps_every_distractor():
    assert len(kept("a rocket launching")) == 80
    assert kept("a dog") == set(sc.COCO) - {"dog", "hot dog"}     # shared word "dog"


def test_softmax_prefers_the_query_when_it_scores_highest():
    query = np.array([[0.30], [0.10]])                             # frame 0 matches, frame 1 does not
    bank = np.full((2, 80), 0.15)
    conf = sc.confidence(query, bank, ["a rocket"], 100.0)
    assert conf.shape == (2, 1) and conf[0, 0] > 0.9 and conf[1, 0] < 0.01


def test_category_fix_raises_confidence_for_a_general_word():
    query = np.array([[0.20]])
    bank = np.full((1, 80), 0.10)
    bank[0, sc.COCO.index("cat")] = 0.25                           # the frame shows a cat
    without_fix = 100.0 * np.r_[query[0], bank[0]]
    plain = np.exp(without_fix - without_fix.max())[0] / np.exp(without_fix - without_fix.max()).sum()
    assert sc.confidence(query, bank, ["an animal"], 100.0)[0, 0] > 0.9 > plain


def frames(*times):
    return [Frame(t, 0, None) for t in times]


def test_frame_cache_embeds_each_timestamp_once():
    calls = []

    def embed(fs):
        calls.append([f.timestamp for f in fs])
        return np.array([[f.timestamp, 1.0] for f in fs])

    first = frame_cache.embed_with_cache(("v", "m"), frames(0.0, 2.0), embed)
    second = frame_cache.embed_with_cache(("v", "m"), frames(2.0, 2.25, 2.0), embed)
    assert calls == [[0.0, 2.0], [2.25]]
    assert first.shape == (2, 2) and np.allclose(second[:, 0], [2.0, 2.25, 2.0])


def test_frame_cache_is_separate_per_video_and_model():
    calls = []

    def embed(fs):
        calls.append(len(fs))
        return np.ones((len(fs), 2))

    for key in [("v", "m"), ("other", "m"), ("v", "other")]:
        frame_cache.embed_with_cache(key, frames(0.0), embed)
    assert calls == [1, 1, 1]


def test_frame_cache_forgets_the_oldest_video(monkeypatch):
    monkeypatch.setattr(frame_cache, "MAX_VIDEOS", 2)
    embed = lambda fs: np.ones((len(fs), 2))
    for name in ["a", "b", "c"]:
        frame_cache.embed_with_cache((name, "m"), frames(0.0), embed)
    assert [k[0] for k in frame_cache._cache] == ["b", "c"]


def test_video_key_depends_on_content(tmp_path):
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"
    a.write_bytes(b"same"), b.write_bytes(b"same"), c.write_bytes(b"different")
    assert frame_cache.video_key(str(a)) == frame_cache.video_key(str(b)) != frame_cache.video_key(str(c))
