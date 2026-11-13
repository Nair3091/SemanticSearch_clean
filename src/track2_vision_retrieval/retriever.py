from typing import List
import cv2
import numpy as np

from src.common.config import load_config
from src.common.types import FrameScores, SubQuery
from src.track2_vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows)
from src.track2_vision_retrieval.embedding import frame_cache
from src.track2_vision_retrieval.scoring import softmax_confidence
from src.track2_vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames


_bank_cache = {}


def _bank_embeddings(model_name: str) -> np.ndarray:
    """Text embeddings of the 80 distractors, computed once per CLIP model."""
    if model_name not in _bank_cache:
        _bank_cache[model_name] = embed_text(softmax_confidence.BANK)
    return _bank_cache[model_name]


def clear_caches() -> None:
    _bank_cache.clear()
    frame_cache.clear()


def _duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return count / fps if fps > 0 else 0.0


def prepare_video(video_path: str) -> dict:
    """Embed a video's frames before any query is known, so a later retrieve() only scores them.
    Always does the Pass-1 frames. A video up to `preindex_full_max_seconds` long also gets the
    whole Pass-2 grid, which covers any windows a query could pick; a longer one gets its Pass-2
    frames when a query needs them. Safe to run alongside retrieve() on the same video."""
    c = load_config()["vision_retrieval"]
    key = (frame_cache.video_key(video_path), c["clip_model_name"])
    duration = _duration(video_path)
    frames = coarse_sample(video_path, c["pass1_interval_seconds"])
    full = duration <= c["preindex_full_max_seconds"]
    if full:
        frames += dense_resample(video_path, 0.0, duration, c["pass2_interval_seconds"])
    frame_cache.embed_with_cache(key, frames, lambda missing: embed_frames(missing))
    _bank_embeddings(c["clip_model_name"])
    return {"duration": duration, "frames": frame_cache.cached_count(key), "full_grid": full}


def retrieve(video_path: str, sub_queries: List[SubQuery]) -> FrameScores:
    if len(sub_queries) == 0:
        raise ValueError("sub_queries must not be empty")
    c = load_config()["vision_retrieval"]
    key = (frame_cache.video_key(video_path), c["clip_model_name"])

    def embed(frames):
        # look up embed_frames at call time so tests can replace it
        return frame_cache.embed_with_cache(key, frames, lambda missing: embed_frames(missing))

    text_emb = embed_text([q.text for q in sub_queries])             # (M, D)
    coarse = coarse_sample(video_path, c["pass1_interval_seconds"])
    coarse_emb = embed(coarse)
    coarse_scores = score_frames(coarse_emb, text_emb)               # (N, M)

    # Window guidance: min over positive sub-queries; if all are negated,
    # frames where the negated concept is weakest are the promising ones.
    positive = [j for j, q in enumerate(sub_queries) if not q.is_negated]
    if positive:
        guide = coarse_scores[:, positive].min(axis=1)
    else:
        guide = -coarse_scores.max(axis=1)

    windows = select_promising_windows(
        coarse, guide, c["top_k_windows"], c["window_radius_seconds"])

    dense = []
    for s, e in windows:
        dense.extend(dense_resample(video_path, s, e, c["pass2_interval_seconds"]))
    dense_emb = embed(dense)

    rows = {}
    for f, e in zip(coarse, coarse_emb):
        rows[round(f.timestamp, 3)] = (e, False)
    for f, e in zip(dense, dense_emb):       # dense overwrites coarse at equal timestamps
        rows[round(f.timestamp, 3)] = (e, True)

    keys = sorted(rows)
    if keys:
        F = np.stack([rows[k][0] for k in keys])
        S = score_frames(F, text_emb)
        bank = score_frames(F, _bank_embeddings(c["clip_model_name"]))
        confidence = softmax_confidence.confidence(S, bank, [q.text for q in sub_queries],
                                                   c["softmax_logit_scale"])
    else:
        S = confidence = np.zeros((0, len(sub_queries)))
    return FrameScores(
        timestamps=np.array(keys),
        scores=[S[:, j] for j in range(S.shape[1])],
        is_dense=np.array([rows[k][1] for k in keys], dtype=bool),
        windows=windows,
        confidence=[confidence[:, j] for j in range(confidence.shape[1])],
    )