"""
Track 2 — Vision & Retrieval

Softmax confidence: how clearly a frame matches the query compared with 80 everyday objects.

Raw CLIP cosine scores sit in a narrow band (about 0.10-0.25), so the gap between a real match
and an absent concept is small and shifts from video to video. Instead, the query competes with
the 80 COCO object names: confidence = softmax([query, distractors] * logit_scale)[query].
This is the scoring used in scripts/score_video.py and scripts/eval_calibration.py.

A distractor that is the same thing as the query, or a kind of it, would steal the probability
from a correct match ("a cat" beats "an animal" on a cat frame). So a distractor is left out when:
  - it shares a content word with the query ("a bus" for "a bus on a street"), or
  - it belongs to a group the query names, using the COCO supercategories plus a few everyday
    words ("animal", "vehicle", "pet", "people", ...).

The raw scores are still what Track 2 uses to find where a moment is; this score is for deciding
whether to accept it.
"""

import re
from typing import Dict, List, Sequence

import numpy as np

# The 80 COCO object categories, grouped by their official COCO supercategory.
SUPERCATEGORIES: Dict[str, List[str]] = {
    "person": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}
COCO = [name for names in SUPERCATEGORIES.values() for name in names]
BANK = [("an " if n[0] in "aeiou" else "a ") + n for n in COCO]

# Everyday words that name a group of COCO objects. A query containing one of these words does
# not compete against any member of that group.
GROUP_WORDS: Dict[str, List[str]] = {
    "person": ["person"], "people": ["person"], "man": ["person"], "men": ["person"],
    "woman": ["person"], "women": ["person"], "boy": ["person"], "girl": ["person"],
    "child": ["person"], "children": ["person"], "kid": ["person"], "someone": ["person"],
    "human": ["person"], "player": ["person"], "rider": ["person"], "pedestrian": ["person"],
    "crowd": ["person"],
    "vehicle": SUPERCATEGORIES["vehicle"], "transport": SUPERCATEGORIES["vehicle"],
    "bike": ["bicycle", "motorcycle"], "motorbike": ["motorcycle"], "plane": ["airplane"],
    "animal": SUPERCATEGORIES["animal"], "creature": SUPERCATEGORIES["animal"],
    "wildlife": SUPERCATEGORIES["animal"], "pet": ["cat", "dog", "bird"],
    "puppy": ["dog"], "kitten": ["cat"], "bag": ["backpack", "handbag", "suitcase"],
    "luggage": ["backpack", "handbag", "suitcase"], "ball": ["sports ball"],
    "sport": SUPERCATEGORIES["sports"], "sports": SUPERCATEGORIES["sports"],
    "utensil": SUPERCATEGORIES["kitchen"], "cutlery": ["fork", "knife", "spoon"],
    "drink": ["bottle", "wine glass", "cup"], "glass": ["wine glass"], "mug": ["cup"],
    "coffee": ["cup"], "tea": ["cup"],
    "food": SUPERCATEGORIES["food"], "meal": SUPERCATEGORIES["food"], "snack": SUPERCATEGORIES["food"],
    "fruit": ["banana", "apple", "orange"], "vegetable": ["broccoli", "carrot"],
    "dessert": ["donut", "cake"],
    "furniture": SUPERCATEGORIES["furniture"], "seat": ["chair", "couch", "bench"],
    "sofa": ["couch"], "table": ["dining table"], "plant": ["potted plant"],
    "electronics": SUPERCATEGORIES["electronic"], "device": SUPERCATEGORIES["electronic"],
    "computer": ["laptop", "keyboard", "mouse"], "phone": ["cell phone"], "television": ["tv"],
    "screen": ["tv", "laptop"],
    "appliance": SUPERCATEGORIES["appliance"], "kitchen": SUPERCATEGORIES["kitchen"] + SUPERCATEGORIES["appliance"],
    "fridge": ["refrigerator"], "toy": ["teddy bear"],
}
_STOP = {"a", "an", "the", "of", "on", "in", "at", "with", "and", "near", "by", "to", "is", "are",
         "some", "any", "his", "her", "their", "its", "into", "from", "under", "over"}


def _words(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _STOP:
            continue
        out.add(w)
        if len(w) > 3 and w.endswith("s"):
            out.add(w[:-1])                       # cars -> car
    return out


def competitors(query: str) -> np.ndarray:
    """Boolean mask over BANK: True for the distractors the query competes against."""
    words = _words(query)
    excluded = set()
    for w in words:
        excluded.update(GROUP_WORDS.get(w, []))
    keep = []
    for name in COCO:
        same_word = bool(_words(name) & words)
        keep.append(not same_word and name not in excluded)
    return np.array(keep, dtype=bool)


def confidence(query_scores: np.ndarray, bank_scores: np.ndarray, queries: Sequence[str],
               logit_scale: float) -> np.ndarray:
    """query_scores: (N, M) raw cosine for M queries. bank_scores: (N, len(BANK)) raw cosine for the
    distractors. Returns (N, M) softmax probability that each query beats its competitors."""
    out = np.zeros_like(query_scores, dtype=np.float32)
    for j, q in enumerate(queries):
        logits = logit_scale * np.concatenate([query_scores[:, j:j + 1], bank_scores[:, competitors(q)]], axis=1)
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        out[:, j] = z[:, 0] / z.sum(axis=1)
    return out
