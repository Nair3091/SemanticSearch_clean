"""
Track 2 — Vision & Retrieval

Reuses CLIP frame embeddings across searches on the same video.

A video is identified by a hash of its file content, so the same clip uploaded twice still hits
the cache. Embeddings are stored per (video, CLIP model, timestamp), which also means a Pass-2
frame that repeats a Pass-1 timestamp is embedded only once.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Callable, List

import numpy as np

from src.common.types import Frame

MAX_VIDEOS = 8                     # oldest video is dropped first

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_locks = {}
_guard = threading.Lock()


def _lock_for(key: tuple) -> threading.Lock:
    """One lock per video: a search waits for a preparation already embedding that video's frames
    instead of embedding them a second time."""
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def video_key(video_path: str) -> str:
    digest = hashlib.sha1()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embed_with_cache(key: tuple, frames: List[Frame],
                     embed: Callable[[List[Frame]], np.ndarray]) -> np.ndarray:
    """Embeds only the frames whose timestamp is not cached yet for `key`, and returns an (N, D)
    array in the same order as `frames`."""
    if not frames:
        return embed([])
    with _lock_for(key):
        with _guard:
            store = _cache.setdefault(key, {})
            _cache.move_to_end(key)
            while len(_cache) > MAX_VIDEOS:
                _cache.popitem(last=False)

        missing, seen = [], set()
        for f in frames:
            t = round(f.timestamp, 3)
            if t not in store and t not in seen:
                missing.append(f)
                seen.add(t)
        if missing:
            for f, emb in zip(missing, embed(missing)):
                store[round(f.timestamp, 3)] = emb
        return np.stack([store[round(f.timestamp, 3)] for f in frames])


def cached_count(key: tuple) -> int:
    with _guard:
        return len(_cache.get(key, {}))


def clear() -> None:
    with _guard:
        _cache.clear()
        _locks.clear()
