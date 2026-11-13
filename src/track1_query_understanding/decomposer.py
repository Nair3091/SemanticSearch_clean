"""
Track 1 — Query Understanding
Owner: Pulavarthi Vishnu Chaitanya

Responsible for:
- Decomposing a natural-language scenario query into atomic sub-queries using Qwen3
- Tagging each sub-query with a negation flag (True/False)
"""

from collections import OrderedDict
from typing import List

from src.common.config import load_config
from src.common.types import SubQuery
from src.track1_query_understanding import track1_qwen
from src.track1_query_understanding.subquery_adapter import structured_to_subqueries

MAX_CACHED_QUERIES = 128
_query_cache: "OrderedDict[tuple, list]" = OrderedDict()


def clear_cache() -> None:
    _query_cache.clear()


def warm_up() -> None:
    """Load Qwen and its prompt cache ahead of the first query (used by the Web App)."""
    track1_qwen.warm_up(load_config()["query_understanding"]["qwen3_model_name"])


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
    key = (settings["qwen3_model_name"], raw_query.strip().lower())
    if key in _query_cache:                      # the same query again: skip Qwen
        _query_cache.move_to_end(key)
        return [SubQuery(text, negated) for text, negated in _query_cache[key]]

    track1_qwen.load_model(settings["qwen3_model_name"])
    result = track1_qwen.understand_query(raw_query)
    sub_queries, _ = structured_to_subqueries(result, raw_query, settings["max_sub_queries"])
    _query_cache[key] = [(q.text, q.is_negated) for q in sub_queries]
    if len(_query_cache) > MAX_CACHED_QUERIES:
        _query_cache.popitem(last=False)
    return sub_queries
