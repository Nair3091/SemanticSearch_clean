"""
Track 1 — Query Understanding
Owner: Pulavarthi Vishnu Chaitanya

Responsible for:
- Decomposing a natural-language scenario query into atomic sub-queries using Qwen3
- Tagging each sub-query with a negation flag (True/False)
"""

from typing import List

from src.common.config import load_config
from src.common.types import SubQuery
from src.track1_query_understanding import track1_qwen
from src.track1_query_understanding.subquery_adapter import structured_to_subqueries


def decompose_query(raw_query: str) -> List[SubQuery]:
    """
    Send the raw NL scenario to Qwen3 and parse the response into
    a list of SubQuery objects.

    Example:
        "a person riding a bike without a helmet"
        -> [SubQuery("a person riding a bike", False),
            SubQuery("a helmet", True)]
    """
    settings = load_config()["query_understanding"]
    track1_qwen.load_model(settings["qwen3_model_name"])
    result = track1_qwen.understand_query(raw_query)
    sub_queries, _ = structured_to_subqueries(result, raw_query, settings["max_sub_queries"])
    return sub_queries
