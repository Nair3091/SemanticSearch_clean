"""
End-to-end pipeline orchestrator: the ONE place that wires Track 1 -> Track 2 -> Track 3.

  query --Qwen3 + adapter--> positive sub-queries + negation captions
        --CLIP two-pass (positives only)--> frame scores
        --min fusion + threshold--> candidate segments
        --ConCLIP filter + final score--> best segment (optionally cut to a clip)

Run from the repo root:
  python scripts/run_pipeline.py --video path/to/video.mp4 --query "a dog without a cat"
  python scripts/run_pipeline.py --video v.mp4 --query "..." --no-qwen             # rule-based Track 1
  python scripts/run_pipeline.py --video v.mp4 --query "..." --skip-negation-check # no ConCLIP weights
  python scripts/run_pipeline.py --video v.mp4 --query "..." --export data/results/clip.mp4

Not wired yet: recovery_layer.py (still stubs), so a low-confidence result is rejected without a retry.
"""

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.config import load_config
from src.common.types import CandidateSegment, RetrievalResult
from src.track1_query_understanding.decomposer import decompose_query
from src.track2_vision_retrieval.retriever import retrieve
from src.track3_fusion_confidence.clip_exporter import export_clip
from src.track3_fusion_confidence.fusion_engine import fuse_final_score, fuse_scores
from src.track3_fusion_confidence.query_adapter import (
    ParsedQuery, find_negation_trigger, from_raw_query, from_sub_queries)
from src.track3_fusion_confidence.rejection import check_confidence

_negation_filter = None


@dataclass
class PipelineReport:
    parsed: ParsedQuery
    frame_scores: Any
    result: RetrievalResult
    candidates: List[dict]            # best first; see select_result for the fields
    fused: Any
    negation_checked: bool
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------- Track 1
def parse_query(query: str, use_qwen: bool = True) -> ParsedQuery:
    """Falls back to rule-based parsing if Qwen fails, or if it misses a negation word in the query."""
    if not use_qwen:
        return from_raw_query(query)
    try:
        sub_queries = decompose_query(query)
    except (ValueError, RuntimeError) as exc:
        parsed = from_raw_query(query)
        parsed.warnings.append(f"Track 1 failed, used rule-based parsing: {str(exc)[:200]}")
        return parsed
    parsed = from_sub_queries(query, sub_queries)
    if not parsed.negated and find_negation_trigger(query):
        rules = from_raw_query(query)
        rules.warnings.append("Qwen missed the negation in this query, used rule-based parsing.")
        return rules
    return parsed


# ---------------------------------------------------------------- Track 3
def build_segments(timestamps: np.ndarray, fused: np.ndarray, windows, score_drop: float,
                   min_length: float) -> list:
    """One candidate per Pass-2 window: the frames around its peak that stay within
    `score_drop` of that peak. The confidence is the peak's fused score."""
    segments = []
    for start, end in windows:
        idx = np.flatnonzero((timestamps >= start) & (timestamps <= end))
        if idx.size == 0:
            continue
        peak = int(np.argmax(fused[idx]))
        floor = fused[idx[peak]] - score_drop
        left = right = peak
        while left > 0 and fused[idx[left - 1]] >= floor:
            left -= 1
        while right < idx.size - 1 and fused[idx[right + 1]] >= floor:
            right += 1
        t0, t1 = float(timestamps[idx[left]]), float(timestamps[idx[right]])
        segments.append(CandidateSegment(t0, max(t1, t0 + min_length), float(fused[idx[peak]])))
    return segments


def select_result(video_path: str, parsed: ParsedQuery, frame_scores, cfg: dict, negation_filter=None):
    """Returns (RetrievalResult, candidates, fused scores). Each candidate is a dict with start,
    end, retrieval_confidence, final_score, status, and for checked segments negation_score,
    pass_ratio and frames_checked. status is one of: accepted, unverified, below_threshold,
    failed_negation."""
    fc = cfg["fusion_confidence"]
    threshold = fc["confidence_threshold"]
    fused = fuse_scores(frame_scores.scores, fc["aggregation"])
    segments = build_segments(frame_scores.timestamps, fused, frame_scores.windows,
                              fc["segment_score_drop"], cfg["vision_retrieval"]["pass2_interval_seconds"])

    def row(seg, status, final=None, **extra):
        return {"start": seg.start_time, "end": seg.end_time, "retrieval_confidence": seg.fused_confidence,
                "final_score": seg.fused_confidence if final is None else final, "status": status,
                "negation_score": None, "pass_ratio": None, "frames_checked": 0, **extra}

    confident = [s for s in segments if check_confidence(s.fused_confidence, threshold)]
    below = [row(s, "below_threshold") for s in segments if s not in confident]

    unverified = parsed.has_negation and negation_filter is None
    if not parsed.has_negation or unverified:
        good = [row(s, "unverified" if unverified else "accepted") for s in confident]
    else:
        good = []
        excluded = [q.text for q in parsed.negated]
        for check in negation_filter.verify_segments(video_path, confident, excluded):
            final = fuse_final_score(check.segment.fused_confidence, check.negation_score,
                                     fc["negation"]["fusion_weight"])
            good.append(row(check.segment, "accepted" if check.passed else "failed_negation", final,
                            negation_score=check.negation_score, pass_ratio=check.pass_ratio,
                            frames_checked=check.frames_checked))

    by_score = lambda c: c["final_score"]
    accepted = sorted((c for c in good if c["status"] != "failed_negation"), key=by_score, reverse=True)
    failed = sorted((c for c in good if c["status"] == "failed_negation"), key=by_score, reverse=True)
    candidates = accepted + failed + sorted(below, key=by_score, reverse=True)

    if accepted:
        best = accepted[0]
        note = "Accepted, but the negation was NOT checked" if unverified else (
            "Accepted, negation check passed" if parsed.has_negation else "Accepted")
        segment = CandidateSegment(best["start"], best["end"], best["final_score"])
        return RetrievalResult(True, segment, note), candidates, fused
    if failed:
        return RetrievalResult(False, None, f"Reject - all {len(failed)} confident candidates "
                                            f"failed the negation check"), candidates, fused
    top = max((s.fused_confidence for s in segments), default=float("nan"))
    return RetrievalResult(False, None, f"Reject - best confidence {top:.3f} is below the "
                                        f"{threshold} threshold"), candidates, fused


def negation_status(cfg: dict):
    """(ready, reason): whether ConCLIP can be loaded on this machine."""
    if importlib.util.find_spec("open_clip") is None:
        return False, "the open_clip package is not installed (pip install open_clip_torch)"
    path = ROOT / cfg["fusion_confidence"]["negation"]["conclip_weights_path"]
    if not path.exists():
        return False, f"ConCLIP weights not found at {path}"
    return True, "ready"


def load_negation_filter(cfg: dict):
    global _negation_filter
    if _negation_filter is None:
        from src.track3_fusion_confidence.negation_filter import ConCLIPNegationFilter

        neg = cfg["fusion_confidence"]["negation"]
        _negation_filter = ConCLIPNegationFilter(
            weights_path=str(ROOT / neg["conclip_weights_path"]),
            model_name=neg["conclip_model_name"],
            frame_threshold=neg["frame_threshold"],
            min_pass_ratio=neg["min_pass_ratio"],
            sample_interval=neg["sample_interval_seconds"],
            max_frames_per_segment=neg["max_frames_per_segment"],
            batch_size=neg["batch_size"],
            device=cfg["hardware"]["device"],
        )
    return _negation_filter


# ---------------------------------------------------------------- all tracks
def run(video_path: str, query: str, use_qwen: bool = True, check_negation: bool = True,
        export_path: Optional[str] = None,
        progress: Optional[Callable[[str, dict], None]] = None) -> PipelineReport:
    """`progress(track, info)` is called as each track starts ({"state": "running"}) and when it
    ends ({"state": "done", "seconds": ...})."""
    cfg = load_config()
    report_progress = progress or (lambda track, info: None)
    timings = {}

    def timed(track, fn):
        report_progress(track, {"state": "running"})
        t0 = time.time()
        value = fn()
        timings[track] = time.time() - t0
        report_progress(track, {"state": "done", "seconds": timings[track]})
        return value

    parsed = timed("track1", lambda: parse_query(query, use_qwen))
    print("Track 1:", parsed.to_dict())

    negation_filter = None
    if parsed.has_negation and check_negation:
        negation_filter = load_negation_filter(cfg)   # fail before the slow retrieval step

    # Negated sub-queries stay out of CLIP retrieval: CLIP would look for the excluded concept.
    frame_scores = timed("track2", lambda: retrieve(video_path, parsed.positive))
    print(f"Track 2: {len(frame_scores.timestamps)} frames scored, windows {frame_scores.windows}")

    result, candidates, fused = timed(
        "track3", lambda: select_result(video_path, parsed, frame_scores, cfg, negation_filter))
    print(f"Track 3: {result.message}")
    for rank, c in enumerate(candidates, 1):
        print(f"  {rank}. {c['start']:.2f}s - {c['end']:.2f}s  score {c['final_score']:.4f}  {c['status']}")

    if result.accepted and export_path:
        method = export_clip(video_path, result.segment.start_time, result.segment.end_time, export_path)
        print(f"Clip written to {export_path} ({method})")
    return PipelineReport(parsed, frame_scores, result, candidates, fused, negation_filter is not None, timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--no-qwen", action="store_true", help="use rule-based query parsing")
    parser.add_argument("--skip-negation-check", action="store_true", help="run without ConCLIP")
    parser.add_argument("--export", help="write the best segment to this .mp4 path")
    args = parser.parse_args()
    run(args.video, args.query, not args.no_qwen, not args.skip_negation_check, args.export)
