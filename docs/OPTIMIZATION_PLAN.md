# Optimization Plan

How to make SemanticSearch faster while keeping all three tracks.

## Done (13 Nov 2026)

Phase 1 is in place, plus the softmax confidence suggested for Track 2. Measured on the same 20-second
test video and laptop CPU:

| Change | Track | Measured |
|---|---|---|
| Frame embeddings reused per video (`embedding/frame_cache.py`), which also covers repeated Pass 2 frames | 2 | First search ~32s; later searches on the same video 0.3–0.6s |
| Qwen prompt prefix processed once and reused (`track1_qwen.py`) | 1 | 134s → 92s and 120s → 93s, identical output |
| Repeated queries skip Qwen (`decomposer.py`) | 1 | Instant |
| Models preload at app start (`Web App/app.py`) | All | 71s for Qwen at startup instead of during the first search |
| Softmax confidence for accept/reject (`scoring/softmax_confidence.py`) | 2 and 3 | 18 of 19 searches correct at 0.25; real matches ≥ 0.208, absent ≤ 0.056 |
| Track 2 processes frames while Qwen parses (`run_pipeline.run`) | 1 and 2 | New query on a new video: 101s → 78s |
| Web App processes the video on upload, while the query is typed (`retriever.prepare_video`) | 2 | New query: 101s → 47s; Quick rules 0.6s |

Raw cosine still locates moments; softmax only decides accept or reject. The sections below are the
original plan; Phase 2 and Phase 3 are still to do.

**Main finding:** nearly all the time goes into two things, Qwen generating text in Track 1 and CLIP ViT-L/14 embedding frames in Track 2. Decoding video, scoring frames and Track 3 are already fast.

## Where the Time Goes Today

Measured on a 20-second, 480p test video on a laptop CPU with no GPU, using 8 threads.

| Track | Step | Time | Worth optimizing? |
|---|---|---|---|
| 1 | Load Qwen3-1.7B | 25s | Once per app start |
| 1 | Qwen parses one query | **131s** for a 191-token answer with the laptop otherwise idle | **Yes, the biggest cost** |
| 2 | Load CLIP from disk cache | 0.35s | No |
| 2 | Encode the query text | 0.02–0.07s | No |
| 2 | Decode all frames | 0.32s | No |
| 2 | CLIP embeds 72 frames | **38s** (about 0.5s per frame) | **Yes** |
| 2 | Score frames against the text | Under 0.01s | No |
| 3 | Load ConCLIP | 2.1s | Once per app start |
| 3 | ConCLIP checks one segment | 0.23s | No |

Other measurements:

- **Timing varies with load:** Track 2 took 38s here but 68–112s in earlier runs on the same video, and Qwen took up to 285s. Those runs shared the laptop with other work. The app uses about 9 GB of the 15.7 GB of RAM with Qwen loaded, so close other apps before timing anything.
- **Resolution barely matters:** 16 frames took 0.34s to decode at 1080p and 0.08s at 480p, and CLIP took about the same time on both (8.7s and 9.2s). Downscaling video first won't help much.
- **Two-pass doesn't help short clips:** with 5 windows of ±3s, the windows covered 15 of 20 seconds of the test video. Pass 1 and Pass 2 scored 72 frames, while a full scan at 0.25s would need 80.
- **Some frames are embedded twice:** 8 of the 62 Pass 2 frames repeat a Pass 1 timestamp.

## Phase 1: No Accuracy Risk

These don't change any scores or decisions, so do them first.

### Track 2: Reuse frame embeddings for each video

Today, every search on the same video embeds all its frames again. Store embeddings by video and timestamp, then reuse them.

- **Effect:** the first search on a video costs the same, and a repeat search that hits the same frames skips about 38s of embedding. Text encoding and scoring take well under a second.
- **Where:** `src/track2_vision_retrieval/retriever.py`, plus a small new cache module. Identify a video by a hash of its file.
- **Also:** keep the cache across runs on disk under `data/frame_cache/`, which is already gitignored.

### Track 2: Stop embedding repeated frames

Skip Pass 2 timestamps that Pass 1 already scored.

- **Effect:** 8 of 72 frames on the test video, about 4s.
- **Where:** `retriever.py`. The embedding cache above handles this automatically.

### Track 1: Reuse results for repeated queries

Keep a small cache of parsed queries.

- **Effect:** a repeated query skips Qwen completely.
- **Where:** `src/track1_query_understanding/decomposer.py`.

### Track 1: Cache the fixed part of the Qwen prompt

The prompt is the same 2,802 tokens every time, with only the query changing at the end. Process it once and reuse the result.

- **Measured:** one query went from **131s to 101s**, and the output was **identical**. A shorter query with a 118-token answer took 57s with the cache; its uncached time wasn't measured. Processing the prompt once costs 55s, so it pays for itself after about two searches.
- **Where:** `src/track1_query_understanding/track1_qwen.py`, in `ask_qwen`. Copy the cache before each query, because generating changes it. Check that it works with the installed `transformers` version.
- **Also:** do this at app start, so users don't pay for it during their first search.

### All tracks: Load models when the app starts

Load Qwen, CLIP and ConCLIP in the background when the Web App starts, so the first search doesn't include 25s or more of loading.

- **Where:** `Web App/app.py`.

## Phase 2: Needs an Accuracy Check

These can change results, so build the labelled test set below first, and only keep a change if accuracy holds.

### Track 1: Use rules first, Qwen when needed

Quick rules are instant and handle queries like `a dog without a cat`. Send a query to Qwen only when it looks complex, such as when it has several actions or words like `then` or `after`.

- **Effect:** simple queries skip Qwen, which saves about 100s or more. Track 1 still does its job, because both paths are part of it.
- **Where:** a small routing function in `decomposer.py`.
- **Risk:** a complex query might get routed to rules. Test both groups of queries.

### Track 1: Make Qwen's output shorter

Generation takes about 0.5s per token, and each answer was 118–191 tokens, so the JSON itself is the main cost. Remove fields the pipeline doesn't use, such as event `attributes` and `relations`.

- **Effect:** estimated at around 30s per query if answers get a third shorter. This hasn't been measured.
- **Where:** `PROMPT_FINAL` in `track1_qwen.py`, and the adapter if needed.
- **Risk:** prompt changes can affect parsing quality. Re-run the Qwen query set.

### Track 2: Use fewer, narrower windows on short clips

Scale `top_k_windows` and `window_radius_seconds` to the video's length.

- **Effect:** for example, 2 windows of ±1.5s scores at most about 36 frames instead of 72, which should roughly halve embedding time on this video.
- **Where:** `retriever.py` and `configs/config.yaml`. `scripts/eval_equivalence.py` already checks whether two-pass finds the same peak as a full scan.
- **Risk:** the right moment could fall outside the windows.

### Track 2: Sample Pass 2 less often

Going from every 0.25s to every 0.5s halves Pass 2.

- **Effect:** about 16s on the test video.
- **Risk:** slightly rougher start and end times, and short moments could be missed.

### Track 2: Test a lighter CLIP model

ViT-B/32 is much faster than ViT-L/14. ConCLIP, which uses the same architecture, took **0.03s per frame, against about 0.5s** for ViT-L/14 on this CPU.

- **Effect:** about 38s down to a few seconds, on the first search too. This is the biggest possible saving in Track 2. The 0.03s was measured on ConCLIP, not on standard B/32.
- **Where:** `clip_model_name` in `configs/config.yaml`.
- **Risk:** it may be less accurate, and `confidence_threshold` must be recalibrated, since its scores use a different scale.

## Phase 3: Hardware and Model Formats

### All tracks: Run on a GPU

This should be the biggest gain, and the code already selects CUDA when it's available. Install a CUDA build of `torch`, since this laptop has the CPU build. Colab or a teammate's GPU laptop would work.

The saved baseline shows about 5ms per frame for embedding, against about 500ms here. That run's model and machine weren't recorded, so treat it only as a sign of what's possible.

### Track 1: Try a quantized or smaller Qwen

A quantized build (for example, GGUF through `llama.cpp`) usually runs faster on a CPU and uses less RAM. `Qwen3-0.6B`, which your original notebook used, is another option. Neither has been measured here, and a smaller model may parse worse, though the adapter and rule fallback do catch some mistakes.

### Track 2: Try an optimized CPU runtime

ONNX Runtime or OpenVINO may speed up CLIP on a CPU with no accuracy loss. This hasn't been measured.

## Not Worth Doing Yet

| Idea | Why not |
|---|---|
| Faster video decoding | Only 0.32s of the total |
| Downscaling video before CLIP | Measured almost no change |
| Using FAISS on short clips | Scoring already takes under 0.01s; it only matters for long videos or many videos |
| Optimizing Track 3 | About 0.23s per segment |
| Sharing frames between Tracks 2 and 3 | Saves about 0.06s |

Track 3 stays as it is, but the recovery layer should reuse the embedding cache when it's built, so retries don't re-embed frames.

## Expected Results

These are **estimates** for one Qwen query on the 20-second test video, on this CPU. Measure them as you go.

| Scenario | Track 1 | Track 2 | Track 3 | Total |
|---|---|---|---|---|
| Today, Qwen | About 130s | About 38s | Under 1s | **About 3 min** |
| Today, Quick rules | Instant | About 38s | Under 1s | **About 40s** |
| Phase 1, new video, Qwen | About 57–101s | About 34s | Under 1s | **About 1.5–2.5 min** |
| Phase 1, repeat search on the same video | Instant if the query repeats | Under 1s | Under 1s | **About 1s** |
| Phase 2, simple query, narrower windows | Instant | About 20s | Under 1s | **About 20s** |
| Phase 2, simple query, ViT-B/32 if accuracy holds | Instant | A few seconds | Under 1s | **Under 10s** |

## How to Keep Accuracy Safe

Speed doesn't help if the answers get worse, and accuracy is already a known issue. Before Phase 2:

1. **Build a labelled test set:** 10–20 real clips, each with queries and correct timestamps. Include queries where the excluded thing is present and where it's absent, plus queries with nothing to find.
2. **Record a baseline:** moment found (does it overlap the correct time?), negation correct, correct rejections and seconds per track.
3. **Change one thing at a time:** re-run the set after each change, and keep it only if accuracy holds.
4. **Recalibrate thresholds:** do this on the same set, and again whenever a model changes. `confidence_threshold` and `frame_threshold` currently come from one video of still photos.

This also gives you real accuracy numbers for your report.

## Suggested Split

| Owner | Work |
|---|---|
| Track 1 (Vishnu) | Query cache, prompt caching, rule-first routing, shorter output, quantized model |
| Track 2 (Dhruv) | Embedding cache, repeated frames, window sizes, Pass 2 interval, ViT-B/32 test |
| Track 3 (Ihsal) | Threshold calibration, the recovery layer, model preloading in the Web App |
| Everyone | The labelled test set and GPU testing |
