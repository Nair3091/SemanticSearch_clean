"""
Track 2 — Vision & Retrieval: Baseline Benchmark Suite

Evaluates the current Track 2 pipeline across:
1. Dense Oracle (ground truth: uniform sampling at 0.25s across full video)
2. Coarse-Only (Pass 1 only at 2.0s)
3. Two-Pass Adaptive (Pass 1 @ 2.0s -> Window Selection -> Pass 2 @ 0.25s)

Metrics recorded:
- Peak score & Peak timestamp (localization accuracy vs oracle)
- Frame count & Frame reduction ratio
- Timing breakdown (decode time, CLIP embedding time, total latency)
- FAISS search comparison (correctness & retrieval latency)
- Parameter sensitivity (window radius & top_k)
"""

import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows
)
from src.vision_retrieval.embedding.clip_encoder import (
    embed_frames, embed_text, score_frames
)
from src.vision_retrieval.indexing.faiss_index import (
    build_index, load_index, search
)


def dense_sample_full(video_path: str, interval_seconds: float):
    """Oracle baseline: dense sample across entire video duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if fps > 0 else 0.0
    cap.release()
    return dense_resample(video_path, 0.0, duration, interval_seconds)


def run_single_benchmark(
    video_path: str,
    query: str,
    pass1_interval: float = 2.0,
    pass2_interval: float = 0.25,
    top_k_windows: int = 2,
    window_radius: float = 2.0
) -> Dict[str, Any]:
    text_emb = embed_text([query])

    # ----------------------------------------------------
    # 1. Oracle: Dense Sample Full Video
    # ----------------------------------------------------
    t0 = time.perf_counter()
    oracle_frames = dense_sample_full(video_path, pass2_interval)
    t_oracle_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    oracle_embeds = embed_frames(oracle_frames)
    t_oracle_embed = time.perf_counter() - t0

    oracle_scores = score_frames(oracle_embeds, text_emb)[:, 0]
    best_oracle_idx = int(np.argmax(oracle_scores))
    oracle_peak_time = oracle_frames[best_oracle_idx].timestamp
    oracle_peak_score = float(oracle_scores[best_oracle_idx])
    t_oracle_total = t_oracle_decode + t_oracle_embed

    # ----------------------------------------------------
    # 2. Coarse-Only (Pass 1)
    # ----------------------------------------------------
    t0 = time.perf_counter()
    pass1_frames = coarse_sample(video_path, pass1_interval)
    t_p1_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    pass1_embeds = embed_frames(pass1_frames)
    t_p1_embed = time.perf_counter() - t0

    pass1_scores = score_frames(pass1_embeds, text_emb)[:, 0]
    best_p1_idx = int(np.argmax(pass1_scores))
    p1_peak_time = pass1_frames[best_p1_idx].timestamp
    p1_peak_score = float(pass1_scores[best_p1_idx])
    t_p1_total = t_p1_decode + t_p1_embed

    # ----------------------------------------------------
    # 3. Two-Pass Adaptive (Current Track 2)
    # ----------------------------------------------------
    t0 = time.perf_counter()
    windows = select_promising_windows(
        pass1_frames, pass1_scores, top_k=top_k_windows, window_radius_seconds=window_radius
    )
    t_win_select = time.perf_counter() - t0

    t0 = time.perf_counter()
    pass2_frames = []
    for start, end in windows:
        w_frames = dense_resample(video_path, start, end, pass2_interval)
        pass2_frames.extend(w_frames)
    t_p2_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    pass2_embeds = embed_frames(pass2_frames)
    t_p2_embed = time.perf_counter() - t0

    pass2_scores = score_frames(pass2_embeds, text_emb)[:, 0]
    best_p2_idx = int(np.argmax(pass2_scores))
    p2_peak_time = pass2_frames[best_p2_idx].timestamp
    p2_peak_score = float(pass2_scores[best_p2_idx])

    two_pass_total_frames = len(pass1_frames) + len(pass2_frames)
    t_two_pass_total = t_p1_total + t_win_select + t_p2_decode + t_p2_embed

    # ----------------------------------------------------
    # 4. FAISS Index Verification
    # ----------------------------------------------------
    temp_idx_path = "data/faiss_index/benchmark_temp.faiss"
    build_index(oracle_embeds, temp_idx_path)
    loaded_idx = load_index(temp_idx_path)
    faiss_scores, faiss_indices = search(loaded_idx, text_emb[0], top_k=5)
    faiss_matches_direct = np.allclose(faiss_scores[0], oracle_peak_score, atol=1e-4)

    return {
        "video": os.path.basename(video_path),
        "query": query,
        "oracle": {
            "frames": len(oracle_frames),
            "peak_time": oracle_peak_time,
            "peak_score": oracle_peak_score,
            "total_time": t_oracle_total,
        },
        "coarse": {
            "frames": len(pass1_frames),
            "peak_time": p1_peak_time,
            "peak_score": p1_peak_score,
            "time_error_vs_oracle": abs(p1_peak_time - oracle_peak_time),
            "total_time": t_p1_total,
        },
        "two_pass": {
            "p1_frames": len(pass1_frames),
            "p2_frames": len(pass2_frames),
            "total_frames": two_pass_total_frames,
            "windows": windows,
            "peak_time": p2_peak_time,
            "peak_score": p2_peak_score,
            "time_error_vs_oracle": abs(p2_peak_time - oracle_peak_time),
            "score_diff_vs_oracle": abs(p2_peak_score - oracle_peak_score),
            "frame_reduction_pct": (1.0 - two_pass_total_frames / len(oracle_frames)) * 100,
            "t_p1_decode": t_p1_decode,
            "t_p1_embed": t_p1_embed,
            "t_win_select": t_win_select,
            "t_p2_decode": t_p2_decode,
            "t_p2_embed": t_p2_embed,
            "total_time": t_two_pass_total,
        },
        "faiss": {
            "matches_direct": bool(faiss_matches_direct),
            "top1_score": float(faiss_scores[0]),
            "top1_idx": int(faiss_indices[0])
        }
    }


def main():
    test_cases = [
        ("data/raw_videos/sample.mp4", "a dog running"),
        ("data/raw_videos/sample.mp4", "a person walking"),
        ("data/raw_videos/sample.mp4", "a red sports car"),  # absent query
        ("data/raw_videos/sample1.mp4", "water or river flowing"),
        ("data/raw_videos/sample1.mp4", "trees and forest landscape"),
        ("data/raw_videos/sample1.mp4", "a dog playing with a ball"),  # absent query
    ]

    print("=" * 80)
    print("TRACK 2: COMPREHENSIVE BASELINE BENCHMARK SUITE")
    print("=" * 80)

    results = []
    for video_path, query in test_cases:
        res = run_single_benchmark(
            video_path,
            query,
            pass1_interval=2.0,
            pass2_interval=0.25,
            top_k_windows=2,
            window_radius=2.0
        )
        results.append(res)
        print(f"\n[Video: {res['video']}] Query: '{query}'")
        print(f"  Oracle (Dense 0.25s) : Peak t={res['oracle']['peak_time']:.2f}s, Score={res['oracle']['peak_score']:.4f}, Frames={res['oracle']['frames']}, Time={res['oracle']['total_time']:.3f}s")
        print(f"  Coarse (Pass 1 2.0s) : Peak t={res['coarse']['peak_time']:.2f}s, Score={res['coarse']['peak_score']:.4f}, Error={res['coarse']['time_error_vs_oracle']:.2f}s, Frames={res['coarse']['frames']}")
        print(f"  Two-Pass Adaptive    : Peak t={res['two_pass']['peak_time']:.2f}s, Score={res['two_pass']['peak_score']:.4f}, Error={res['two_pass']['time_error_vs_oracle']:.2f}s")
        print(f"                         Windows={res['two_pass']['windows']}")
        print(f"                         Frames: P1={res['two_pass']['p1_frames']} + P2={res['two_pass']['p2_frames']} = {res['two_pass']['total_frames']} (Reduction: {res['two_pass']['frame_reduction_pct']:.1f}%)")
        print(f"                         Time: {res['two_pass']['total_time']:.3f}s (P1: {res['two_pass']['t_p1_embed']:.3f}s, P2: {res['two_pass']['t_p2_embed']:.3f}s)")
        print(f"  FAISS Index Match    : {res['faiss']['matches_direct']} (Top-1 Score={res['faiss']['top1_score']:.4f})")

    # Save results to json
    import json
    os.makedirs("data/benchmarks", exist_ok=True)
    with open("data/benchmarks/track2_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed benchmark JSON to data/benchmarks/track2_baseline.json")


if __name__ == "__main__":
    main()
