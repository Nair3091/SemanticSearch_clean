"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Query adapter: turns Track 1's output into what Track 2 and Track 3 need.

Track 1 can hand over its output in two ways, and both are supported:

  (a) structured_query.json  -- what Vishnu's Qwen3 notebook writes today:
        {
          "original_query": "...",
          "structured_query": {
              "entities": [...],
              "events": [{"event_id", "action", "subject", "object",
                          "attributes", "negated"}, ...],
              "negations": [{"event_id", "trigger"}, ...],
              ...
          },
          "negation_verification": [...]
        }

  (b) List[SubQuery]  -- what src.track1_query_understanding.decomposer.decompose_query()
      will return once it is implemented (SubQuery.text is the POSITIVE phrasing,
      SubQuery.is_negated says whether it must be ABSENT).

Either way this module produces a ParsedQuery with:
  - positive sub-queries  -> sent to Track 2 (CLIP retrieval). Negated
    sub-queries are NOT sent to CLIP retrieval, because vanilla CLIP ignores the
    negation word and would actively retrieve the thing the user wants excluded.
  - negation captions     -> sent to Track 3's ConCLIP verifier.

This file does not import anything from Track 1; it only reads Track 1's
output format, so Track 1 can change its internals freely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from src.common.types import SubQuery


# Word-boundary negation triggers (longest first). Word boundaries matter:
# a plain substring check would flag "no" inside "snow" / "know" / "piano".
NEGATION_TRIGGERS = [
    "did not", "does not", "do not", "is not", "are not", "was not", "were not",
    "didn't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",
    "devoid of", "free of", "but not",
    "without", "excluding", "except", "lacking", "absent",
    "never", "none", "not", "no",
]
_TRIGGER_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(NEGATION_TRIGGERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Max words for using the whole raw query as the ConCLIP caption.
# CLIP's text encoder truncates at 77 tokens; ~30 words stays safely under that.
_MAX_WORDS_FOR_RAW_CAPTION = 30


@dataclass
class ParsedQuery:
    raw_query: str
    positive: List[SubQuery] = field(default_factory=list)
    negated: List[SubQuery] = field(default_factory=list)
    negation_captions: List[str] = field(default_factory=list)
    source: str = ""            # "structured_json" | "sub_queries" | "rule_based"
    warnings: List[str] = field(default_factory=list)

    @property
    def has_negation(self) -> bool:
        return len(self.negation_captions) > 0

    @property
    def all_sub_queries(self) -> List[SubQuery]:
        return self.positive + self.negated

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "source": self.source,
            "positive_sub_queries": [s.text for s in self.positive],
            "negated_sub_queries": [s.text for s in self.negated],
            "negation_captions": self.negation_captions,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def find_negation_trigger(text: str) -> Optional[str]:
    m = _TRIGGER_RE.search(text or "")
    return m.group(1).lower() if m else None


def strip_negation(text: str) -> str:
    """'did not get inside' -> 'get inside', 'without a helmet' -> 'a helmet'."""
    out = _TRIGGER_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", out).strip()


def _clean(*parts: Optional[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _event_phrase(event: dict) -> str:
    """'woman' + 'picked up' + 'black bag' (+ attributes) -> one CLIP-friendly phrase."""
    subject = str(event.get("subject") or "")
    attrs = event.get("attributes") or []
    if attrs:
        subject = _clean(subject, "wearing" if subject else "", ", ".join(map(str, attrs)))
    return _clean(subject, str(event.get("action") or ""), str(event.get("object") or ""))


def _raw_query_usable_as_caption(raw_query: str, n_negations: int) -> bool:
    return n_negations == 1 and 0 < len(raw_query.split()) <= _MAX_WORDS_FOR_RAW_CAPTION


# ---------------------------------------------------------------------------
# (a) Track 1 structured JSON
# ---------------------------------------------------------------------------

def from_structured_json(data: dict) -> ParsedQuery:
    """Build a ParsedQuery from Vishnu's structured_query.json dict."""
    raw_query = str(data.get("original_query") or "")
    sq = data.get("structured_query", data)  # tolerate the inner dict being passed directly
    raw_query = raw_query or str(sq.get("original_query") or "")

    events = sq.get("events") or []
    negated_ids = {n.get("event_id") for n in (sq.get("negations") or [])}
    triggers = {n.get("event_id"): n.get("trigger") for n in (sq.get("negations") or [])}

    pq = ParsedQuery(raw_query=raw_query, source="structured_json")

    neg_events = []
    for ev in events:
        phrase = _event_phrase(ev)
        if not phrase:
            continue
        is_neg = bool(ev.get("negated")) or ev.get("event_id") in negated_ids
        if is_neg:
            # positive phrasing (what must be ABSENT) for SubQuery.text
            pq.negated.append(SubQuery(text=strip_negation(phrase), is_negated=True))
            neg_events.append((ev, phrase))
        else:
            pq.positive.append(SubQuery(text=phrase, is_negated=False))

    # ConCLIP captions
    if _raw_query_usable_as_caption(raw_query, len(neg_events)):
        # Short single-negation query: the user's own sentence is the best caption
        # (it matches the NegBench caption style ConCLIP was evaluated on).
        pq.negation_captions = [raw_query]
    else:
        for ev, phrase in neg_events:
            trig = triggers.get(ev.get("event_id"))
            if find_negation_trigger(phrase) is None:
                # Qwen marked it negated but the action text lost the negation word
                phrase = _clean(str(ev.get("subject") or ""), trig or "not",
                                str(ev.get("action") or ""), str(ev.get("object") or ""))
            pq.negation_captions.append(phrase)

    # Edge case: Track 1's own verifier flagged "REVIEW" (a negation word is in the
    # query but Qwen did not attach it to any event). Don't silently ignore it.
    review = [v for v in (data.get("negation_verification") or []) if v.get("verification") == "REVIEW"]
    if review and not neg_events:
        pq.warnings.append("Track 1 flagged a negation it could not assign to an event; "
                           "falling back to rule-based negation parsing.")
        fallback = from_raw_query(raw_query)
        pq.negated.extend(fallback.negated)
        pq.negation_captions.extend(fallback.negation_captions)

    if not pq.positive:
        # e.g. "no people" -> nothing positive to retrieve on; use the de-negated query
        pq.positive.append(SubQuery(text=strip_negation(raw_query) or raw_query, is_negated=False))
        pq.warnings.append("No positive sub-query from Track 1; using the raw query for retrieval.")

    return pq


def load_structured_json(path: "str | Path") -> ParsedQuery:
    with open(path, "r", encoding="utf-8") as f:
        return from_structured_json(json.load(f))


# ---------------------------------------------------------------------------
# (b) Track 1 decompose_query() output
# ---------------------------------------------------------------------------

def from_sub_queries(raw_query: str, sub_queries: Iterable[SubQuery]) -> ParsedQuery:
    sub_queries = list(sub_queries)
    pq = ParsedQuery(raw_query=raw_query, source="sub_queries")
    pq.positive = [s for s in sub_queries if not s.is_negated]
    pq.negated = [s for s in sub_queries if s.is_negated]

    if _raw_query_usable_as_caption(raw_query, len(pq.negated)):
        pq.negation_captions = [raw_query]
    else:
        context = pq.positive[0].text if pq.positive else ""
        for s in pq.negated:
            pq.negation_captions.append(_clean(context, "without", s.text) if context else _clean("no", s.text))

    if not pq.positive:
        pq.positive.append(SubQuery(text=strip_negation(raw_query) or raw_query, is_negated=False))
        pq.warnings.append("No positive sub-query from Track 1; using the raw query for retrieval.")
    return pq


# ---------------------------------------------------------------------------
# Fallback: no Track 1 output available (e.g. Qwen not installed on this machine)
# ---------------------------------------------------------------------------

def from_raw_query(raw_query: str) -> ParsedQuery:
    """
    Rule-based fallback so the pipeline can still run end-to-end without Qwen3.
    """
    pq = ParsedQuery(raw_query=raw_query, source="rule_based")
    m = _TRIGGER_RE.search(raw_query)
    if not m:
        pq.positive = [SubQuery(text=raw_query.strip(), is_negated=False)]
        return pq

    before = raw_query[:m.start()].strip(" ,.;")
    after = raw_query[m.end():].strip(" ,.;")
    # "with no cars" -> drop the dangling "with"
    before = re.sub(r"\b(with|and|but|that has|which has|that are|are|is|that|which|who)\b$", "", before, flags=re.IGNORECASE).strip(" ,")

    # If the user typed a purely negative query like "no humans"
    if not before:
        pq.positive = [SubQuery(text="a scene", is_negated=False)]
    else:
        pq.positive = [SubQuery(text=before, is_negated=False)]
        
    if after:
        pq.negated = [SubQuery(text=after, is_negated=True)]
        pq.negation_captions = [raw_query.strip()]
    return pq
