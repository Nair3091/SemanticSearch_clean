"""Wiring tests for scripts/run_pipeline.py. The sampler and retriever are real; CLIP, ConCLIP and
Qwen are replaced by small stand-ins, so these tests say nothing about retrieval quality."""
import copy

import cv2
import numpy as np
import pytest

from scripts import run_pipeline as rp
from src.common.config import load_config
from src.common.types import FrameScores, SubQuery
from src.track1_query_understanding import track1_qwen
from src.track2_vision_retrieval import retriever
from src.track3_fusion_confidence.negation_filter import SegmentNegationResult
from src.track3_fusion_confidence.query_adapter import from_sub_queries

CFG = {
    "vision_retrieval": {"pass1_interval_seconds": 2.0, "pass2_interval_seconds": 0.25,
                         "top_k_windows": 2, "window_radius_seconds": 1.0},
    "fusion_confidence": {"aggregation": "min", "confidence_threshold": 0.5, "segment_score_drop": 0.02,
                          "negation": {"fusion_weight": 0.5}},
}


class FakeFilter:
    """Fails any segment that starts in `bad_ranges`, and records what it was asked."""

    def __init__(self, bad_ranges=()):
        self.bad_ranges, self.captions = bad_ranges, None

    def verify_segments(self, video, segments, captions):
        self.captions = captions
        out = []
        for seg in segments:
            bad = any(a <= seg.start_time <= b for a, b in self.bad_ranges)
            out.append(SegmentNegationResult(seg, "FAIL" if bad else "PASS", 0.4, 0.0 if bad else 1.0, 4))
        return out


def scores(ts, values, windows):
    ts = np.array(ts, dtype=float)
    return FrameScores(ts, [np.array(values, dtype=np.float32)], np.ones(len(ts), dtype=bool), windows)


def parsed(negated=False):
    subs = [SubQuery("a dog", False)] + ([SubQuery("a cat", True)] if negated else [])
    return from_sub_queries("a dog without a cat" if negated else "a dog", subs)


TWO_MOMENTS = dict(ts=[0, 1, 5, 6], values=[0.6, 0.1, 0.1, 0.8], windows=[(0, 1), (5, 6)])


def two_moments():
    return scores(TWO_MOMENTS["ts"], TWO_MOMENTS["values"], TWO_MOMENTS["windows"])


def test_segment_grows_around_peak_and_stops_at_a_drop():
    ts, fused = np.array([0, 1, 2, 3, 4.0]), np.array([0.1, 0.59, 0.6, 0.59, 0.1])
    seg = rp.build_segments(ts, fused, [(0, 4)], 0.02, 0.25)[0]
    assert (seg.start_time, seg.end_time) == (1.0, 3.0) and seg.fused_confidence == pytest.approx(0.6)


def test_single_frame_segment_gets_a_real_length():
    seg = rp.build_segments(np.array([0, 1, 2.0]), np.array([0.1, 0.6, 0.1]), [(0, 2)], 0.02, 0.25)[0]
    assert seg.end_time - seg.start_time == 0.25


def test_low_confidence_is_rejected_but_closest_moment_is_kept():
    result, candidates, _ = rp.select_result("v", parsed(), scores([0, 1], [0.1, 0.2], [(0, 1)]), CFG)
    assert not result.accepted and result.segment is None and "Reject" in result.message
    assert candidates[0]["status"] == "below_threshold" and candidates[0]["start"] == 1.0


def test_no_negation_returns_best_segment():
    result, candidates, _ = rp.select_result("v", parsed(), two_moments(), CFG)
    assert result.accepted and result.segment.start_time == 6.0
    assert [c["status"] for c in candidates] == ["accepted", "accepted"]


def test_negation_failure_removes_the_top_segment():
    f = FakeFilter(bad_ranges=[(5, 7)])
    result, candidates, _ = rp.select_result("v", parsed(True), two_moments(), CFG, f)
    assert result.accepted and result.segment.start_time == 0.0
    assert f.captions == ["a cat"]               # the concept itself, not the whole sentence
    assert result.segment.fused_confidence == pytest.approx(0.5 * 0.6 + 0.5 * 0.4)
    assert [c["status"] for c in candidates] == ["accepted", "failed_negation"]


def test_all_segments_failing_negation_is_a_rejection():
    result, candidates, _ = rp.select_result("v", parsed(True), two_moments(), CFG, FakeFilter([(0, 9)]))
    assert not result.accepted and "negation" in result.message
    assert all(c["status"] == "failed_negation" for c in candidates)


def test_skipped_negation_check_is_stated_in_the_result():
    result, candidates, _ = rp.select_result("v", parsed(True), two_moments(), CFG, None)
    assert result.accepted and "NOT checked" in result.message
    assert candidates[0]["status"] == "unverified"


def test_qwen_failure_falls_back_to_rules(monkeypatch):
    def boom(_):
        raise ValueError("bad json")

    monkeypatch.setattr(track1_qwen, "load_model", lambda name: (name, "cpu"))
    monkeypatch.setattr(track1_qwen, "understand_query", boom)
    pq = rp.parse_query("a dog without a cat")
    assert pq.source == "rule_based" and pq.has_negation and "Track 1 failed" in pq.warnings[0]


def fake_qwen(monkeypatch, structured):
    monkeypatch.setattr(track1_qwen, "load_model", lambda name: (name, "cpu"))
    monkeypatch.setattr(track1_qwen, "understand_query", lambda q: {"structured_query": structured})


def test_qwen_misparse_still_flows_through_both_adapters(monkeypatch):
    fake_qwen(monkeypatch, {
        "entities": [{"type": "object", "name": "dog", "attributes": []},
                     {"type": "object", "name": "cat", "attributes": []}],
        "events": [{"event_id": 1, "action": "has", "subject": "dog", "object": "cat", "negated": False}]})
    pq = rp.parse_query("a dog without a cat")
    assert [s.text for s in pq.positive] == ["a dog"] and [s.text for s in pq.negated] == ["a cat"]


def test_negation_lost_by_qwen_and_adapter_falls_back_to_rules(monkeypatch):
    fake_qwen(monkeypatch, {
        "entities": [{"type": "object", "name": "kitchen", "attributes": []}], "events": []})
    pq = rp.parse_query("a kitchen excluding any people")
    assert pq.source == "rule_based" and [s.text for s in pq.negated] == ["any people"]
    assert "missed the negation" in pq.warnings[0]


def test_word_containing_no_does_not_trigger_the_fallback(monkeypatch):
    fake_qwen(monkeypatch, {"entities": [{"type": "object", "name": "snow", "attributes": []}], "events": []})
    pq = rp.parse_query("snow")
    assert pq.source == "sub_queries" and not pq.has_negation


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.avi")   # 10s @ 10fps, gets brighter over time
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(100):
        w.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    w.release()
    return path


def test_full_run_only_sends_positives_to_retrieval(video, monkeypatch):
    seen, progress = [], []

    def fake_embed_frames(frames):
        out = []
        for f in frames:
            p = float(f.image.mean()) / 255.0
            v = np.array([p, 1.0 - p], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def fake_embed_text(texts):
        seen.extend(texts)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    monkeypatch.setattr(retriever, "embed_frames", fake_embed_frames)
    monkeypatch.setattr(retriever, "embed_text", fake_embed_text)
    monkeypatch.setattr(retriever, "load_config", lambda: CFG)
    monkeypatch.setattr(rp, "load_config", lambda: {**CFG, "hardware": {"device": "cpu"}})
    monkeypatch.setattr(rp, "load_negation_filter", lambda cfg: FakeFilter())

    report = rp.run(video, "a bright room without a lamp", use_qwen=False,
                    progress=lambda track, info: progress.append((track, info["state"])))
    assert seen == ["a bright room"]             # "a lamp" never reaches CLIP retrieval
    assert report.result.accepted and report.result.segment.end_time > 7.0
    assert report.negation_checked
    assert progress == [(t, s) for t in ("track1", "track2", "track3") for s in ("running", "done")]


def test_missing_conclip_weights_stop_the_run_before_retrieval(video, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["fusion_confidence"]["negation"]["conclip_weights_path"] = "data/conclip/does_not_exist.pt"
    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    monkeypatch.setattr(rp, "_negation_filter", None)
    monkeypatch.setattr(rp, "retrieve", lambda *a: pytest.fail("retrieval should not start"))
    assert rp.negation_status(cfg)[0] is False
    with pytest.raises(FileNotFoundError):
        rp.run(video, "a dog without a cat", use_qwen=False)
