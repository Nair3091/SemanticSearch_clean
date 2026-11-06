import numpy as np
import pytest

from src.common.types import CandidateSegment, SubQuery
from src.track3_fusion_confidence.fusion_engine import fuse_final_score, fuse_scores
from src.track3_fusion_confidence.negation_filter import decide_segment, segment_sample_times
from src.track3_fusion_confidence.query_adapter import find_negation_trigger, from_raw_query, from_sub_queries
from src.track3_fusion_confidence.rejection import check_confidence, split_by_confidence


def test_fuse_takes_weakest_sub_query():
    out = fuse_scores([np.array([0.3, 0.1]), np.array([0.2, 0.4])])
    assert np.allclose(out, [0.2, 0.1])


def test_fuse_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        fuse_scores([np.array([0.1])], "median")


def test_final_score_ignores_weight_without_negation():
    assert fuse_final_score(0.3, None, 0.5) == 0.3
    assert fuse_final_score(0.3, 0.1, 0.5) == pytest.approx(0.2)


@pytest.mark.parametrize("score,expected", [(0.28, True), (0.27, False), (None, False), (float("nan"), False)])
def test_check_confidence(score, expected):
    assert check_confidence(score, 0.28) is expected


def test_split_by_confidence():
    good, bad = CandidateSegment(0, 1, 0.3), CandidateSegment(1, 2, 0.1)
    assert split_by_confidence([good, bad], 0.2) == ([good], [bad])


def test_decide_segment_passes_when_concept_scores_are_low():
    sims = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.3], [0.1, 0.1]])   # one concept visible in 1 of 4 frames
    verdict, score, ratio, per_frame = decide_segment(sims, 0.20, 0.7)
    assert verdict == "PASS" and ratio == 0.75 and per_frame[2] == pytest.approx(0.3)
    assert score == pytest.approx(1 - 0.15)
    assert decide_segment(sims, 0.20, 0.8)[0] == "FAIL"
    assert decide_segment(np.zeros((0, 1)), 0.20, 0.7)[0] == "ERROR"


def test_decide_segment_fails_when_the_concept_is_visible():
    assert decide_segment(np.array([[0.25], [0.26], [0.24]]), 0.205, 0.7)[0] == "FAIL"


def test_segment_sample_times():
    assert segment_sample_times(0.0, 2.0, 1.0, 16) == [0.0, 1.0, 2.0]
    assert len(segment_sample_times(0.0, 100.0, 1.0, 16)) == 16
    assert segment_sample_times(5.0, 5.0, 1.0, 16) == [5.0]


def test_triggers_match_whole_words_only():
    assert find_negation_trigger("snow on a notebook") is None
    assert find_negation_trigger("a dog without a cat") == "without"


def test_raw_query_fallback_splits_negation():
    pq = from_raw_query("a street with no cars")
    assert [s.text for s in pq.positive] == ["a street"]
    assert [s.text for s in pq.negated] == ["cars"]
    assert pq.negation_captions == ["a street with no cars"]


def test_sub_queries_separate_positive_from_negated():
    subs = [SubQuery("a dog", False), SubQuery("a cat", True)]
    pq = from_sub_queries("a dog without a cat", subs)
    assert [s.text for s in pq.positive] == ["a dog"]
    assert pq.negation_captions == ["a dog without a cat"]


def test_two_negations_get_one_caption_each():
    subs = [SubQuery("a dog", False), SubQuery("a cat", True), SubQuery("a bird", True)]
    pq = from_sub_queries("a dog without a cat or a bird", subs)
    assert pq.negation_captions == ["a dog without a cat", "a dog without a bird"]
