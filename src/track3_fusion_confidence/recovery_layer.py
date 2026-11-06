"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Responsible for:
- Widening the search window / lowering Pass-1 threshold on low confidence
- Dropping weak sub-queries
- Triggering a single re-scoring retry back through the processing layer
"""

from typing import List
from src.common.types import SubQuery


def relax_query(sub_queries: List[SubQuery]) -> List[SubQuery]:
    """
    Drop the weakest sub-query/queries to relax the search constraints.
    """
    raise NotImplementedError


def widen_window(current_threshold: float, current_interval: float) -> tuple:
    """
    Lower the Pass-1 sampling threshold / widen the sampling window
    for a single retry pass.
    Returns (new_threshold, new_interval).
    """
    raise NotImplementedError
