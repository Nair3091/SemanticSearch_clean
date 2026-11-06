"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Responsible for:
- Fusing per-sub-query scores into a single confidence score per candidate segment
  (min-aggregation across sub-queries, per the design doc)
- Fusing the retrieval confidence (Track 2, CLIP) with the negation score
  (Track 3, ConCLIP) into the final ranking score
"""

from typing import List, Optional, Union
import numpy as np


def fuse_scores(sub_query_scores: Union[List[np.ndarray], np.ndarray],
                aggregation: str = "min") -> np.ndarray:
    """
    Combine per-sub-query score arrays into one fused confidence score
    per candidate segment, using min-aggregation.

    Accepts either:
      - a list of M arrays, each of shape (N,)  -> one array per sub-query
      - a single (N, M) matrix, exactly what clip_encoder.score_frames returns
        (rows = frames, columns = sub-queries)

    Why min? A frame is only as good as its weakest constraint. If the query is
    "a man in a red shirt riding a bike", a frame that matches "riding a bike"
    perfectly but has no man in a red shirt should NOT score highly. Mean would
    hide that; min exposes it.

    Returns a (N,) float32 array.
    """
    if isinstance(sub_query_scores, np.ndarray):
        matrix = sub_query_scores
        if matrix.ndim == 1:
            return matrix.astype(np.float32)
    else:
        if len(sub_query_scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        matrix = np.stack([np.asarray(s, dtype=np.float32) for s in sub_query_scores], axis=1)

    if matrix.ndim != 2:
        raise ValueError(f"Expected an (N, M) score matrix, got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        # No positive sub-queries at all -> nothing to rank on
        return np.zeros((matrix.shape[0],), dtype=np.float32)

    if aggregation == "min":
        fused = matrix.min(axis=1)
    elif aggregation == "mean":
        fused = matrix.mean(axis=1)
    elif aggregation == "max":
        fused = matrix.max(axis=1)
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}' (use min / mean / max)")

    return fused.astype(np.float32)


def fuse_final_score(retrieval_confidence: float,
                     negation_score: Optional[float],
                     negation_weight: float) -> float:
    """
    Final score used to rank the clips shown to the user.

    - No negation in the query  -> final score = retrieval confidence (Track 2).
    - Negation in the query     -> weighted fusion:
          final = (1 - w) * retrieval_confidence + w * negation_score
      where negation_score is 1 - the mean ConCLIP similarity of the segment to the
      excluded concept(s), so it is higher when the concept is more clearly absent.
      A change in similarity moves the score by the same amount as a change in
      retrieval confidence, so the two can be summed without re-normalisation.
    """
    if negation_score is None:
        return float(retrieval_confidence)
    w = float(np.clip(negation_weight, 0.0, 1.0))
    return float((1.0 - w) * retrieval_confidence + w * negation_score)
