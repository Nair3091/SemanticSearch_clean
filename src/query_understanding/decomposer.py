"""
Track 1 — Query Understanding
Owner: Pulavarthi Vishnu Chaitanya

Responsible for:
- Decomposing a natural-language scenario query into atomic sub-queries using Qwen3
- Tagging each sub-query with a negation flag (True/False)
"""

from typing import List

from src.common.types import SubQuery


def decompose_query(raw_query: str) -> List[SubQuery]:
    """
    Send the raw NL scenario to Qwen3 and parse the response into
    a list of SubQuery objects.

    Example:
        "a person riding a bike without a helmet"
        -> [SubQuery("person riding a bike", False),
            SubQuery("person wearing a helmet", True)]
    """
    raise NotImplementedError
