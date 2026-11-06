"""Adapter: Track 1 structured query  ->  Track 2 sub-queries (List[SubQuery]).

Track 1 returns entities / events / negations as JSON. Track 2 (retrieve()) takes a list of
SubQuery(text, is_negated), where a negated sub-query carries the POSITIVE text ("a cat") plus
the flag, because CLIP cannot encode "no cat" (it scores it like "cat"). Track 3 inverts it.

Rules (deliberately simple and printed back as a trace so they can be inspected):
  1. Each event becomes one sub-query: "<subject np> <action> <object np>", where a noun
     phrase is "a/an" + the entity's attributes + its name, e.g. "a tall man entering a white car".
     The event's `negated` flag becomes is_negated.
  2. Each entity that no event mentions becomes its own positive sub-query ("a red hat").
     If a negation word sits directly in front of it in the original query ("without a cat",
     "but not a bird"), it becomes a NEGATED sub-query instead. Track 1 only records negation
     on events, so this covers "a dog without a cat"-style queries. It is a heuristic.
  2b. If a NON-negated event mentions an entity that the query negates this way, Track 1
     missed the negation word (Qwen3-1.7B turned "a dog without a cat" into "dog has cat").
     That event is dropped and its entities fall through to rule 2.
  2c. Qwen sometimes stores the negation word as an attribute ("without" on "helmet") or invents
     an entity the query never mentions ("a road"). The word is removed from the attributes and
     the entity counts as negated; an invented entity that no event uses is dropped.
  3. If nothing could be extracted, fall back to the raw query as one positive sub-query.
  4. At most `max_sub_queries` are kept (config: query_understanding.max_sub_queries),
     always keeping at least one positive.
Not used (yet): temporal relations, event attributes, relations.
"""
import re
from typing import List, Tuple

from src.common.types import SubQuery

NEGATION_WORDS = {"no", "not", "without", "never", "excluding", "exclude", "nor"}
_MISSING = {"", "null", "none", "n/a", "na", "nil"}
_FILLER = {"a", "an", "the", "any", "some", "other"}


def _clean(value):
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in _MISSING else s


def _noun_phrase(name: str, attributes) -> str:
    attrs = [str(a).strip() for a in (attributes or [])
             if str(a).strip() and str(a).strip().lower() not in NEGATION_WORDS]
    words = " ".join(attrs + [name])
    article = "an" if words[:1].lower() in "aeiou" else "a"
    return f"{article} {words}"


def _negated_in_query(query: str, name: str, attributes) -> bool:
    """True if a negation word governs the entity in the query. Only articles, the entity's own
    attributes and -ing verbs may sit in between ("without a small red hat", "not wearing a hat")."""
    words = re.findall(r"[a-z0-9']+", query.lower())
    target = name.lower().split()
    allowed = set(_FILLER)
    for a in attributes or []:
        allowed.update(str(a).lower().split())
    allowed -= NEGATION_WORDS
    n = len(target)
    for i in range(len(words) - n + 1):
        if words[i:i + n] != target:
            continue
        j = i - 1
        while j >= 0 and (words[j] in allowed or words[j].endswith("ing")):
            j -= 1
        if j >= 0 and words[j] in NEGATION_WORDS:
            return True
    return False


def structured_to_subqueries(structured, original_query: str,
                             max_sub_queries: int = 5) -> Tuple[List[SubQuery], List[str]]:
    """Returns (sub_queries, trace). `structured` is Track 1's structured_query dict
    (or the full understand_query() result, which is unwrapped)."""
    if isinstance(structured, dict) and "structured_query" in structured:
        structured = structured["structured_query"]
    structured = structured or {}

    ents = {}
    for e in structured.get("entities", []) or []:
        if isinstance(e, dict) and _clean(e.get("name")):
            ents[str(e["name"]).strip().lower()] = e

    # entities with a negation word directly in front of them in the query
    query_negated = {
        name_l for name_l, e in ents.items()
        if _negated_in_query(original_query, str(e["name"]).strip(), e.get("attributes"))
    }

    def phrase(name):
        name = _clean(name)
        if name is None:
            return None
        ent = ents.get(name.lower())
        return _noun_phrase(name, ent.get("attributes") if ent else [])

    out: List[SubQuery] = []
    trace: List[str] = []
    seen = set()

    def add(text, negated, why):
        key = (text.lower(), negated)
        if key in seen:
            return
        seen.add(key)
        out.append(SubQuery(text, negated))
        trace.append(f"{why} -> \"{text}\" ({'NEGATED' if negated else 'positive'})")

    covered = set()
    for ev in structured.get("events", []) or []:
        if not isinstance(ev, dict):
            continue
        subj, obj = _clean(ev.get("subject")), _clean(ev.get("object"))
        action = _clean(ev.get("action"))
        ev_negated = bool(ev.get("negated", False))
        hit = [n for n in (subj, obj) if n and n.lower() in query_negated]
        if hit and not ev_negated:
            # Track 1 missed the negation word: the event is a misparse, so the entities
            # fall through to the entity rules below.
            trace.append(f"event {ev.get('event_id', '?')} dropped: not marked negated, "
                         f"but the query negates {hit}")
            continue
        for n in (subj, obj):
            if n:
                covered.add(n.lower())
        sp, op = phrase(subj), phrase(obj)
        if sp and op and not action:
            text = f"{sp} and {op}"
        else:
            text = " ".join(p for p in (sp, action, op) if p)
        if not text:
            continue
        add(text, ev_negated, f"event {ev.get('event_id', '?')}")

    for name_l, ent in ents.items():
        if name_l in covered:
            continue
        name = str(ent["name"]).strip()
        attrs = ent.get("attributes") or []
        if name_l not in original_query.lower():
            trace.append(f"entity '{name}' dropped: not in the query")
            continue
        negated = name_l in query_negated
        why = (f"entity '{name}' (not in any event, negation word in query)" if negated
               else f"entity '{name}' (not in any event)")
        add(_noun_phrase(name, attrs), negated, why)

    if not out:
        out.append(SubQuery(original_query.strip(), False))
        trace.append(f"nothing extracted -> raw query \"{original_query.strip()}\" (positive)")

    if len(out) > max_sub_queries:
        kept = out[:max_sub_queries]
        first_pos = next((q for q in out if not q.is_negated), None)
        if first_pos is not None and not any(not q.is_negated for q in kept):
            kept[-1] = first_pos
        dropped = [q.text for q in out if q not in kept]
        trace.append(f"capped at {max_sub_queries}; dropped: {dropped}")
        out = kept
    return out, trace