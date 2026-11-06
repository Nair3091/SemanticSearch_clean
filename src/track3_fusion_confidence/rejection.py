"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Responsible for:
- Evaluating the fused confidence score against a calibrated threshold
- Deciding: accept result, or flag for the recovery layer / final reject
"""

import math
from typing import List, Tuple

from src.common.types import CandidateSegment


def check_confidence(fused_score: float, threshold: float) -> bool:
    """
    Returns True if the fused score clears the calibrated threshold
    (i.e. the result can be accepted / returned to the user).

    NaN / None scores are always rejected (a broken score should never be
    shown to a user as a confident result).
    """
    if fused_score is None:
        return False
    try:
        score = float(fused_score)
    except (TypeError, ValueError):
        return False
    if math.isnan(score):
        return False
    return score >= threshold


def split_by_confidence(segments: List[CandidateSegment],
                        threshold: float) -> Tuple[List[CandidateSegment], List[CandidateSegment]]:
    """
    Partition candidate segments into (accepted, rejected) using check_confidence.
    The recovery layer is triggered by the orchestrator when `accepted` is empty.
    """
    accepted, rejected = [], []
    for seg in segments:
        (accepted if check_confidence(seg.fused_confidence, threshold) else rejected).append(seg)
    return accepted, rejected
