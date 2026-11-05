"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Responsible for:
- Fusing per-sub-query scores into a single confidence score per candidate segment
  (min-aggregation across sub-queries, per the design doc)
"""

from typing import List
import numpy as np


def fuse_scores(sub_query_scores: List[np.ndarray]) -> np.ndarray:
    """
    Combine per-sub-query score arrays into one fused confidence score
    per candidate segment, using min-aggregation.
    """
    raise NotImplementedError
