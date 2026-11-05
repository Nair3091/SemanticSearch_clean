from typing import List
import numpy as np

from src.common.config import load_config
from src.common.types import FrameScores, SubQuery
from src.vision_retrieval.sampling.two_pass_sampler import (
    coarse_sample, dense_resample, select_promising_windows)
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames


def retrieve(video_path: str, sub_queries: List[SubQuery]) -> FrameScores:
    if len(sub_queries) == 0:
        raise ValueError("sub_queries must not be empty")
    c = load_config()["vision_retrieval"]

    text_emb = embed_text([q.text for q in sub_queries])             # (M, D)
    coarse = coarse_sample(video_path, c["pass1_interval_seconds"])
    coarse_scores = score_frames(embed_frames(coarse), text_emb)     # (N, M)

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
    dense_scores = score_frames(embed_frames(dense), text_emb)

    rows = {}
    for f, r in zip(coarse, coarse_scores):
        rows[round(f.timestamp, 3)] = (r, False)
    for f, r in zip(dense, dense_scores):    # dense overwrites coarse at equal timestamps
        rows[round(f.timestamp, 3)] = (r, True)

    keys = sorted(rows)
    S = np.stack([rows[k][0] for k in keys]) if keys else np.zeros((0, len(sub_queries)))
    return FrameScores(
        timestamps=np.array(keys),
        scores=[S[:, j] for j in range(S.shape[1])],
        is_dense=np.array([rows[k][1] for k in keys], dtype=bool),
        windows=windows,
    )