"""Importable wrapper around Vishnu's Track 1 notebook export (DEMO USE ONLY).

The notebook export loads the model, calls input() and runs the query at import time, so it
cannot be imported. This module keeps his logic and prompt VERBATIM (everything from
parse_json_response down to the end of PROMPT_FINAL is a straight copy of lines 137-1155 of
his file) and only changes how the model is loaded and called:

  * the model loads lazily via load_model(), not at import
  * no trust_remote_code (Qwen3 is supported natively by transformers)
  * input is NOT truncated; the notebook capped input at 4096 tokens with truncation on, and
    because the query is appended at the END of a ~13.5k-character prompt, a prompt over the
    cap would silently lose the query. ask_qwen prints the real token count and warns above 4096.
  * understand_query() does not write structured_query.json

When Vishnu delivers a proper module, replace this file and keep the adapter.
"""
import gc
import json
import re
import time

SYSTEM_MESSAGE = (
    "You are a query understanding module who can understand conversations and situations. "
    "Return only one valid JSON object and no Markdown."
)
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 600
MAX_INPUT_TOKENS = 8192      # hard stop; the notebook used 4096 with silent truncation
NOTEBOOK_INPUT_CAP = 4096

_state = {"model": None, "tokenizer": None, "device": None, "name": None, "last_raw": None}


def load_model(model_name=DEFAULT_MODEL):
    """Load tokenizer + model once. fp16 on CUDA, fp32 on CPU (same as the notebook)."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, low_cpu_mem_usage=True)
    except (TypeError, ValueError):   # older transformers only know torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _state.update(model=model, tokenizer=tokenizer, device=device, name=model_name)
    return model_name, device


def ask_qwen(prompt, max_new_tokens=MAX_NEW_TOKENS, verbose=True):
    import torch
    model, tokenizer, device = _state["model"], _state["tokenizer"], _state["device"]
    if model is None:
        raise RuntimeError("Call load_model() first.")

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(formatted_text, return_tensors="pt")          # no truncation
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens > MAX_INPUT_TOKENS:
        raise ValueError(f"Prompt is {input_tokens} tokens, over the {MAX_INPUT_TOKENS} limit.")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    generated = outputs[0][input_tokens:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    _state["last_raw"] = response
    if verbose:
        warn = (f"  WARNING: over the notebook's {NOTEBOOK_INPUT_CAP}-token cap (it would have truncated the query)"
                if input_tokens > NOTEBOOK_INPUT_CAP else "")
        print(f"  [qwen] {input_tokens} input tokens, {len(generated)} new tokens, {elapsed:.1f}s{warn}")

    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return response


# ---------------------------------------------------------------------------------------------
# VERBATIM COPY of Vishnu's code (lines 137-1155 of track1_query_understanding_and_polishing_.py)
# ---------------------------------------------------------------------------------------------
def parse_json_response(response):
    cleaned = response.strip()
    cleaned = re.sub(r"^\\s*```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s*```\\s*$", "", cleaned)

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    preview = cleaned[:1000]
    raise ValueError(
        "Qwen did not return a valid JSON object. Response preview:\\n" + preview
    )

NEGATION_TRIGGERS = [
    "did not",
    "does not",
    "do not",
    "didn't",
    "doesn't",
    "don't",
    "without",
    "never",
    "but not",
    "excluding",
    "exclude",
    "not",
    "no"
]

def find_negation_triggers(query):
    query_lower = query.lower()
    found = []

    for trigger in NEGATION_TRIGGERS:
        if trigger in query_lower:
            found.append(trigger)

    return found

def verify_negation(query, structured_result):
    detected_triggers = find_negation_triggers(query)

    negation_events = {
        item.get("event_id"): item.get("trigger")
        for item in structured_result.get("negations", [])
    }

    verification_results = []

    for event in structured_result.get("events", []):
        event_id = event.get("event_id")
        qwen_negated = event.get("negated", False)

        expected_negated = event_id in negation_events
        trigger = negation_events.get(event_id)

        if qwen_negated == expected_negated:
            status = "VALID"
        else:
            status = "CORRECTED"

        if expected_negated:
            event["negated"] = True

        verification_results.append({
            "event_id": event_id,
            "action": event.get("action"),
            "original_negated": qwen_negated,
            "verified_negated": event.get("negated", False),
            "negation_trigger": trigger,
            "verification": status
        })

    if detected_triggers and not negation_events:
        verification_results.append({
            "verification": "REVIEW",
            "detected_triggers": detected_triggers,
            "message": "Negation detected in query but Qwen did not assign it to an event."
        })

    return verification_results

import re

VALID_ENTITY_TYPES = {
    "person",
    "object",
    "vehicle",
    "clothing",
    "location"
}

def extract_fallback_attributes(query, entity_name):
    """
    Rule-based fallback to extract adjectives preceding the entity name
    from the original query in case the smaller LLM misses them in the JSON attributes.
    """
    query_lower = query.lower()
    name_lower = entity_name.lower()

    # Common descriptors, colors, and modifiers
    potential_attributes = [
        "white", "black", "red", "blue", "green", "yellow", "brown", "grey", "gray",
        "hair", "haired", "tall", "short", "young", "old", "large", "small", "little",
        "big", "leather", "denim", "cotton", "woolen", "dark", "light"
    ]

    extracted = []
    # Check if any potential descriptive words immediately precede or are near the entity name in the query
    words = query_lower.split()
    try:
        idx = words.index(name_lower)
        for lookback in range(max(0, idx - 3), idx):
            word = words[lookback].strip(".,!?;:")
            if word in potential_attributes:
                extracted.append(word)
    except ValueError:
        # If not found directly, check general substring occurrence
        for attr in potential_attributes:
            if f"{attr} {name_lower}" in query_lower or f"{attr}-{name_lower}" in query_lower:
                extracted.append(attr)

    return list(set(extracted))

def verify_entities(query, structured_result):
    entities = structured_result.get("entities", [])
    verified_entities = []
    seen = set()

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "")).strip().lower()

        if not name:
            continue

        key = name.lower()
        if key in seen:
            continue

        seen.add(key)

        attributes = entity.get("attributes", [])
        if not isinstance(attributes, list):
            attributes = [str(attributes)]

        attributes = [
            str(attribute).strip()
            for attribute in attributes
            if str(attribute).strip()
        ]

        # Fallback attribute extraction if the LLM left the attributes array empty
        if not attributes:
            attributes = extract_fallback_attributes(query, name)

        # Dynamically accept the model's assigned type instead of overwriting with hardcoded keywords.
        # If the LLM assigned an unknown type category, we gracefully default it to "object".
        if entity_type not in VALID_ENTITY_TYPES:
            entity_type = "object"

        verified_entities.append({
            "type": entity_type,
            "name": name,
            "attributes": attributes
        })

    structured_result["entities"] = verified_entities
    return structured_result

TEMPORAL_PATTERNS = [
    "followed by",
    "after that",
    "and then",
    "then",
    "later",
    "after",
    "before",
    "next",
    "finally",
    "first"
]

def verify_temporal_relations(query, structured_result):
    events = structured_result.get("events", [])

    if len(events) < 2:
        structured_result["temporal_relations"] = []
        return structured_result

    query_lower = query.lower()

    explicit_temporal = any(
        pattern in query_lower
        for pattern in TEMPORAL_PATTERNS
    )

    if not explicit_temporal:
        structured_result["temporal_relations"] = []
        return structured_result

    event_ids = [
        event.get("event_id")
        for event in events
        if event.get("event_id") is not None
    ]

    relations = []

    def add_relation(event1, event2):
        if event1 == event2:
            return

        relation = {
            "relation": "before",
            "event1": event1,
            "event2": event2
        }

        if relation not in relations:
            relations.append(relation)

    # Qwen's explicit temporal relations are trusted first.
    qwen_relations = structured_result.get("temporal_relations", [])

    if isinstance(qwen_relations, list):
        for relation in qwen_relations:
            if not isinstance(relation, dict):
                continue

            if relation.get("relation") != "before":
                continue

            event1 = relation.get("event1")
            event2 = relation.get("event2")

            if event1 in event_ids and event2 in event_ids:
                add_relation(event1, event2)

    # If Qwen did not produce relations but the query explicitly
    # indicates a simple sequence, preserve event order.
    if not relations and any(
        marker in query_lower
        for marker in ["then", "and then", "followed by", "finally"]
    ):
        for i in range(len(event_ids) - 1):
            add_relation(event_ids[i], event_ids[i + 1])

    structured_result["temporal_relations"] = relations

    return structured_result

PROMPT_FINAL = """
You are the Query Understanding module of a video semantic search system.

Convert the user's natural-language video search query into ONE structured JSON object for a visual retrieval system.

The goal is to preserve all explicitly stated information while keeping entities, attributes, events, relationships, temporal order, and negations clearly separated.

CORE PRINCIPLE:

ENTITY = the actual person, object, vehicle, clothing item, or location.

ATTRIBUTE = a word or phrase that describes an entity.

Always separate the main entity from its descriptive words.

Examples:
"red top hat"
-> entity: hat
-> attributes: ["red", "top"]

"black leather jacket"
-> entity: jacket
-> attributes: ["black", "leather"]

"small blue bag"
-> entity: bag
-> attributes: ["small", "blue"]

"white car"
-> entity: car
-> attributes: ["white"]

Never use "red top hat" as the entity name when "hat" is the actual entity.

==================================================
1. ENTITY EXTRACTION
==================================================

Extract every explicitly mentioned:

- person
- object
- vehicle
- clothing item
- location

Each entity must appear exactly once in the "entities" array.

Use the most specific entity type:

person:
- man
- woman
- boy
- girl
- person
- child

vehicle:
- car
- truck
- bus
- van
- motorcycle
- bicycle

clothing:
- hat
- shirt
- jacket
- coat
- dress
- pants
- trousers
- shoes
- etc.

object:
- bag
- phone
- bottle
- ball
- umbrella
- etc.

location:
- road
- street
- building
- park
- room
- shop
- etc.

==================================================
2. ENTITY NAMES
==================================================

The "name" must contain ONLY the core entity.

Remove descriptive adjectives from the entity name.

Examples:

"red hat"
-> name: "hat"

"red top hat"
-> name: "hat"

"black leather jacket"
-> name: "jacket"

"large black bag"
-> name: "bag"

"white sedan"
-> name: "sedan"

"small white car"
-> name: "car"

Do NOT create:

"name": "red hat"

Do NOT create:

"name": "red top hat"

The descriptive words belong in attributes.

==================================================
3. ATTRIBUTE EXTRACTION
==================================================

Extract every explicitly stated adjective or descriptive phrase that describes an entity.

Attributes must always be represented as a JSON array of strings.

Examples:

"red top hat"
-> entity:
{
    "type": "clothing",
    "name": "hat"
}
-> attributes:
["red", "top"]

"black leather jacket"
-> entity:
{
    "type": "clothing",
    "name": "jacket"
}
-> attributes:
["black", "leather"]

"large red bag"
-> entity:
{
    "type": "object",
    "name": "bag"
}
-> attributes:
["large", "red"]

"white car"
-> entity:
{
    "type": "vehicle",
    "name": "car"
}
-> attributes:
["white"]

Preserve ALL explicitly stated attributes.

Do not discard adjectives.

Do not invent attributes.

Do not infer attributes.

==================================================
4. ATTRIBUTE OWNERSHIP
==================================================

Every attribute must belong to the entity it explicitly describes.

Example:

"A man wearing a red top hat entered a white car."

Entities:

man
hat
car

Attributes:

hat -> ["red", "top"]
car -> ["white"]

The event should preserve these attributes when the entity participates in the event.

==================================================
5. PEOPLE
==================================================

A person can have attributes.

Example:

"A tall man wearing a black coat entered a car."

Person:
{
    "type": "person",
    "name": "man"
}

Person attributes:
["tall"]

Clothing:
{
    "type": "clothing",
    "name": "coat"
}

Clothing attributes:
["black"]

The clothing itself is an entity, but "wearing" does NOT automatically become a separate event.

==================================================
6. CLOTHING
==================================================

When clothing describes a person performing an action, do NOT create a separate wearing event.

Example:

"A man wearing a black jacket entered a car."

Correct:

Entity:
man

Entity:
jacket
attribute: black

Event:
man entered car

The jacket can be represented through the event's attributes or relations.

Do NOT create:

Event:
man wearing jacket

unless the query explicitly treats the wearing action itself as an event.

==================================================
7. OBJECTS
==================================================

Objects must be separated from their descriptive attributes.

Example:

"red ball"

Correct:
entity = ball
attributes = ["red"]

Example:

"small black bag"

Correct:
entity = bag
attributes = ["small", "black"]

Never merge adjectives into the entity name.

==================================================
8. VEHICLES
==================================================

Vehicles must use the entity type "vehicle".

Examples:

"white car"
-> type: vehicle
-> name: car
-> attributes: ["white"]

"large blue truck"
-> type: vehicle
-> name: truck
-> attributes: ["large", "blue"]

"black motorcycle"
-> type: vehicle
-> name: motorcycle
-> attributes: ["black"]

==================================================
9. EVENTS
==================================================

Separate distinct actions into separate events.

Every event MUST use exactly this structure:

{
    "event_id": 1,
    "action": "...",
    "subject": "...",
    "object": "...",
    "attributes": [],
    "relations": [],
    "negated": false
}

Use the entity's CORE NAME in subject and object.

Example:

"A man wearing a red hat entered a white car."

Event:

{
    "event_id": 1,
    "action": "entering",
    "subject": "man",
    "object": "car",
    "attributes": ["red hat", "white"],
    "relations": [],
    "negated": false
}

Do not invent actions.

==================================================
10. SHARED SUBJECTS
==================================================

If multiple events refer to the same person, use the same subject.

Example:

"A man entered a car and then walked away."

Events:

Event 1:
subject = man
action = entering
object = car

Event 2:
subject = man
action = walking
object = null

==================================================
11. RELATIONSHIPS
==================================================

Relationships describe how entities are connected.

Examples:

wearing
carrying
holding
standing near
behind
in front of
next to
inside
near
with

Use relations when the relationship is explicitly stated.

Example:

"A man carrying a bag walked."

The relationship can be represented as:

"relations": [
    {
        "relation": "carrying",
        "subject": "man",
        "object": "bag"
    }
]

Do not invent relationships.

==================================================
12. TEMPORAL ORDER
==================================================

Preserve the order of explicitly sequential events.

Temporal indicators include:

then
after
before
followed by
next
first
later
finally

If two events are explicitly sequential, create a temporal relation.

Example:

"A woman picked up a bag and then entered a car."

Events:

1 = picked up bag
2 = entered car

Temporal relation:

{
    "relation": "before",
    "event1": 1,
    "event2": 2
}

==================================================
13. TEMPORAL RULES
==================================================

Only create temporal relationships when the query explicitly indicates temporal order.

Do NOT assume temporal order merely because one event appears before another in the sentence.

"but", "however", and "although" are NOT temporal indicators.

==================================================
14. NEGATION
==================================================

Explicit negation must be preserved.

Negation indicators include:

not
no
without
never
did not
does not
do not
didn't
doesn't
but not
excluding

A negated action must have:

"negated": true

Example:

"A man did not enter the car."

Event:

{
    "event_id": 1,
    "action": "entering",
    "subject": "man",
    "object": "car",
    "attributes": [],
    "relations": [],
    "negated": true
}

==================================================
15. NEGATION SEPARATION
==================================================

Never hide a negated condition inside the attributes of a positive event.

Example:

"A man ate an apple without wearing a hat."

Correct:

Event 1:
man ate apple
negated = false

Event 2:
man wearing hat
negated = true

Negation:

{
    "event_id": 2,
    "trigger": "without"
}

Do NOT put "not wearing hat" inside event 1 attributes.

==================================================
16. EXPLICIT INFORMATION ONLY
==================================================

Only extract information explicitly present in the query.

Do NOT infer:

- gender from clothing
- color from context
- object identity from assumptions
- actions that are not stated
- locations that are not stated
- relationships that are not stated
- attributes that are not stated

==================================================
17. ENTITY UNIQUENESS
==================================================

Each explicitly mentioned entity should appear only once in "entities".

If the same entity is mentioned multiple times, reuse the same entity.

Do not create duplicate entities.

Example:

"A man entered a car. The man left the car."

Entities:

man
car

NOT:

man
car
man
car

==================================================
18. ENTITY TYPE CONSISTENCY
==================================================

The same entity must always keep the same type.

Do not assign multiple types to the same entity.

Example:

"car"

must remain:

{
    "type": "vehicle",
    "name": "car"
}

Do not later classify it as an object.

==================================================
19. ATTRIBUTE CONSISTENCY
==================================================

If an entity is described with attributes, preserve those attributes consistently.

Example:

"A red car was parked. The car was entered by a man."

The car remains:

name = car
attributes = ["red"]

==================================================
20. LOCATIONS
==================================================

Explicitly mentioned locations must appear as entities.

Example:

"A man entered a shop."

Entities:

man
shop

==================================================
21. NO EMPTY ENTITY NAMES
==================================================

Never create:

"name": ""

Every entity must have a valid name.

==================================================
22. NO INVENTION
==================================================

Never add information that does not exist in the query.

Do not create additional entities, attributes, events, relationships, timestamps, or temporal relationships.

==================================================
23. NO TIMESTAMPS
==================================================

Do not identify timestamps.

Do not estimate timestamps.

==================================================
24. NO VISUAL RETRIEVAL
==================================================

Only understand and structure the query.

Do not perform visual retrieval.

==================================================
25. JSON FORMAT
==================================================

Return ONLY valid JSON.

No markdown.

No explanations.

No comments.

No text outside the JSON object.

The output MUST follow this structure:

{
    "original_query": "...",
    "entities": [
        {
            "type": "person",
            "name": "...",
            "attributes": []
        },
        {
            "type": "object",
            "name": "...",
            "attributes": []
        },
        {
            "type": "vehicle",
            "name": "...",
            "attributes": []
        },
        {
            "type": "clothing",
            "name": "...",
            "attributes": []
        },
        {
            "type": "location",
            "name": "...",
            "attributes": []
        }
    ],
    "events": [
        {
            "event_id": 1,
            "action": "...",
            "subject": "...",
            "object": "...",
            "attributes": [],
            "relations": [],
            "negated": false
        }
    ],
    "temporal_relations": [
        {
            "relation": "before",
            "event1": 1,
            "event2": 2
        }
    ],
    "negations": [
        {
            "event_id": 2,
            "trigger": "not"
        }
    ]
}

==================================================
26. FINAL CHECK BEFORE OUTPUT
==================================================

Before returning the JSON, verify:

1. Every explicitly mentioned entity exists in "entities".
2. Every entity appears only once.
3. Entity names contain the core noun, not its adjectives.
4. Every explicit adjective is stored in "attributes".
5. "red top hat" becomes:
   name = "hat"
   attributes = ["red", "top"]
6. "black leather jacket" becomes:
   name = "jacket"
   attributes = ["black", "leather"]
7. Vehicles use type "vehicle".
8. Clothing uses type "clothing".
9. Distinct actions are separate events.
10. Shared subjects remain consistent.
11. Explicit temporal words create temporal relations.
12. Negated conditions have "negated": true.
13. Every negated event is recorded in "negations".
14. No unstated information is added.
15. The result is valid JSON.
16. Return ONLY the JSON object.

Return ONLY the JSON object.
"""

# ---------------------------------------------------------------------------------------------
# End of verbatim copy.
# ---------------------------------------------------------------------------------------------
def understand_query(query):
    """Same pipeline as the notebook's understand_query(), minus the file write."""
    prompt = PROMPT_FINAL + "\n\nUser Query:\n" + query
    response = ask_qwen(prompt, max_new_tokens=MAX_NEW_TOKENS)
    result = parse_json_response(response)
    result = verify_entities(query, result)
    result = verify_temporal_relations(query, result)
    verification = verify_negation(query, result)
    return {
        "original_query": query,
        "structured_query": result,
        "negation_verification": verification,
    }
