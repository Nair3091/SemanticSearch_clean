from src.common.types import SubQuery
from src.integration.subquery_adapter import structured_to_subqueries


def _texts(sqs):
    return [(q.text, q.is_negated) for q in sqs]


def test_event_with_attributes_from_entities():
    # prompt example: "A tall man wearing a black coat entered a white car."
    s = {"entities": [{"type": "person", "name": "man", "attributes": ["tall"]},
                      {"type": "clothing", "name": "coat", "attributes": ["black"]},
                      {"type": "vehicle", "name": "car", "attributes": ["white"]}],
         "events": [{"event_id": 1, "action": "entering", "subject": "man", "object": "car",
                     "attributes": [], "relations": [], "negated": False}]}
    sq, _ = structured_to_subqueries(s, "A tall man wearing a black coat entered a white car")
    # event -> one sub-query; coat is in no event -> its own positive sub-query
    assert _texts(sq) == [("a tall man entering a white car", False), ("a black coat", False)]


def test_negated_event_keeps_positive_text_and_flag():
    # prompt example: "A man ate an apple without wearing a hat."
    s = {"entities": [{"type": "person", "name": "man", "attributes": []},
                      {"type": "object", "name": "apple", "attributes": []},
                      {"type": "clothing", "name": "hat", "attributes": []}],
         "events": [{"event_id": 1, "action": "eating", "subject": "man", "object": "apple", "negated": False},
                    {"event_id": 2, "action": "wearing", "subject": "man", "object": "hat", "negated": True}]}
    sq, _ = structured_to_subqueries(s, "A man ate an apple without wearing a hat")
    assert _texts(sq) == [("a man eating an apple", False), ("a man wearing a hat", True)]


def test_entities_only_query_no_events():
    s = {"entities": [{"type": "object", "name": "dog", "attributes": []}], "events": []}
    sq, _ = structured_to_subqueries(s, "a dog")
    assert _texts(sq) == [("a dog", False)]


def test_entity_negated_by_word_in_query():
    s = {"entities": [{"type": "object", "name": "dog", "attributes": []},
                      {"type": "object", "name": "cat", "attributes": []}], "events": []}
    sq, _ = structured_to_subqueries(s, "a dog without a cat")
    assert _texts(sq) == [("a dog", False), ("a cat", True)]


def test_negation_word_before_attributes_and_only_directly_before():
    s = {"entities": [{"type": "object", "name": "dog", "attributes": []},
                      {"type": "clothing", "name": "hat", "attributes": ["small", "red"]}], "events": []}
    assert _texts(structured_to_subqueries(s, "a dog without a small red hat")[0]) == \
        [("a dog", False), ("a small red hat", True)]
    # "not" is not directly in front of "hat" here (a verb sits in between) -> stays positive
    assert _texts(structured_to_subqueries(s, "a dog not wearing a small red hat")[0]) == \
        [("a dog", False), ("a small red hat", False)]


def test_null_strings_and_missing_object():
    s = {"entities": [{"type": "person", "name": "man", "attributes": []}],
         "events": [{"event_id": 1, "action": "walking", "subject": "man", "object": "null", "negated": False}]}
    sq, _ = structured_to_subqueries(s, "a man walking")
    assert _texts(sq) == [("a man walking", False)]


def test_article_an_and_attribute_order():
    s = {"entities": [{"type": "object", "name": "umbrella", "attributes": ["large", "red"]}], "events": []}
    assert _texts(structured_to_subqueries(s, "x")[0]) == [("a large red umbrella", False)]
    s = {"entities": [{"type": "object", "name": "apple", "attributes": []}], "events": []}
    assert _texts(structured_to_subqueries(s, "x")[0]) == [("an apple", False)]


def test_fallback_to_raw_query_when_empty():
    sq, trace = structured_to_subqueries({"entities": [], "events": []}, "  something odd ")
    assert _texts(sq) == [("something odd", False)] and "raw query" in trace[0]


def test_unwraps_full_understand_query_result():
    full = {"original_query": "a dog", "negation_verification": [],
            "structured_query": {"entities": [{"type": "object", "name": "dog", "attributes": []}], "events": []}}
    assert _texts(structured_to_subqueries(full, "a dog")[0]) == [("a dog", False)]


def test_duplicates_removed_and_cap_keeps_a_positive():
    ents = [{"type": "object", "name": n, "attributes": []} for n in ["a1", "b1", "c1", "d1"]]
    s = {"entities": ents, "events": []}
    sq, trace = structured_to_subqueries(s, "x", max_sub_queries=2)
    assert len(sq) == 2 and "capped" in trace[-1]
    # cap must not leave only negated sub-queries when a positive exists
    s = {"entities": [{"type": "object", "name": "dog", "attributes": []}],
         "events": [{"event_id": i, "action": "x", "subject": f"s{i}", "object": None, "negated": True}
                    for i in range(3)]}
    sq, _ = structured_to_subqueries(s, "dog", max_sub_queries=3)
    assert any(not q.is_negated for q in sq)


def test_missed_negation_event_is_replaced():
    # Qwen3-1.7B parsed "a dog without a cat" as one non-negated event "dog has cat"
    s = {"entities": [{"type": "object", "name": "dog", "attributes": []},
                      {"type": "object", "name": "cat", "attributes": []}],
         "events": [{"event_id": 1, "action": "has", "subject": "dog", "object": "cat",
                     "negated": False}]}
    sq, trace = structured_to_subqueries(s, "a dog without a cat")
    assert _texts(sq) == [("a dog", False), ("a cat", True)]
    assert "dropped" in trace[0]