"""Compare Track 1 Qwen backends (transformers fp32 vs llama.cpp GGUF) on speed and parse quality.

  py scripts/bench_qwen_backends.py hf                                   -> fp32 baseline
  py scripts/bench_qwen_backends.py data/models/Qwen3-1.7B-Q4_K_M.gguf   -> llama.cpp
Writes data/benchmarks/bench_qwen_<name>.json; compare the files with --compare a.json b.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track1_query_understanding import track1_qwen
from src.track1_query_understanding.subquery_adapter import structured_to_subqueries

QUERIES = [
    "a dog looking at a crab",
    "a dog without a cat",
    "a person riding a bike without a helmet",
    "a woman drinking coffee without wearing glasses",
    "A man wearing a red top hat entered a white car",
    "a woman picked up a bag and then entered a car",
    "a man in a black jacket walking a dog but not on a leash",
    "two kids playing football in a park and then a dog runs onto the field",
]


def run(backend):
    if backend == "hf":
        track1_qwen.load_model(track1_qwen.DEFAULT_MODEL)
        track1_qwen.warm_up(track1_qwen.DEFAULT_MODEL)
    else:
        track1_qwen.load_model(backend)
        track1_qwen.warm_up(backend)
    rows = []
    for q in QUERIES:
        t = time.time()
        try:
            result = track1_qwen.understand_query(q)
            subs, _ = structured_to_subqueries(result, q, 5)
            subs = [[s.text, s.is_negated] for s in subs]
            err = None
        except Exception as e:  # noqa: BLE001 - record and keep going
            subs, err = None, f"{type(e).__name__}: {e}"[:300]
        dt = time.time() - t
        print(f"{dt:6.1f}s  {q!r} -> {subs or err}", flush=True)
        rows.append({"query": q, "seconds": round(dt, 2), "sub_queries": subs, "error": err,
                     "raw": track1_qwen._state.get("last_raw")})
    name = "hf" if backend == "hf" else os.path.splitext(os.path.basename(backend))[0]
    out = f"data/benchmarks/bench_qwen_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out)


def compare(a_path, b_path):
    a, b = (json.load(open(p, encoding="utf-8")) for p in (a_path, b_path))
    same = 0
    for ra, rb in zip(a, b):
        match = ra["sub_queries"] == rb["sub_queries"]
        same += match
        print(f"{'SAME' if match else 'DIFF'}  {ra['seconds']:6.1f}s vs {rb['seconds']:6.1f}s  {ra['query']}")
        if not match:
            print("      A:", ra["sub_queries"] or ra["error"])
            print("      B:", rb["sub_queries"] or rb["error"])
    ta, tb = sum(r["seconds"] for r in a), sum(r["seconds"] for r in b)
    print(f"\nidentical sub-queries: {same}/{len(a)}   total {ta:.0f}s vs {tb:.0f}s ({ta / tb:.1f}x)")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
