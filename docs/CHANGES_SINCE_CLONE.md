# Changes Since the First Clone

This compares the repo as first cloned (commit `dfc8f1e`, "clean repo commit") with the current state.

- **Commits on `integration/three-tracks-web-app`:** `cac5454` (all three tracks and the Web App), `6d2dfd5` (README and these docs), then the optimization round in section 11. `main` is still at `dfc8f1e`.

To see the code differences yourself:

```bash
git diff -M dfc8f1e HEAD
```

## Summary

| Area | At first clone | Now |
|---|---|---|
| Folders in `src/` | `query_understanding`, `integration`, `vision_retrieval`, `fusion_confidence`, `recovery`, `webapp` | `track1_query_understanding`, `track2_vision_retrieval`, `track3_fusion_confidence`, plus `common` |
| Track 1 | `decompose_query` was a stub; working code was a demo-only wrapper | `decompose_query` works, using Qwen3-1.7B and the adapter |
| Track 2 | Working | Same logic; only import paths changed |
| Track 3 | Stubs | Real fusion, rejection, ConCLIP negation check and clip export |
| `run_pipeline.py` | Stopped at the Track 1 stub | Runs all three tracks |
| Web demo | Commented-out stub | Working app in `Web App/` |
| Tests | 28 | 67 |

## 1. Folder Restructure

### Files moved

| Before | After |
|---|---|
| `src/integration/track1_qwen.py` | `src/track1_query_understanding/track1_qwen.py` |
| `src/integration/subquery_adapter.py` | `src/track1_query_understanding/subquery_adapter.py` |
| `src/query_understanding/decomposer.py` | `src/track1_query_understanding/decomposer.py` |
| `src/vision_retrieval/` (all files) | `src/track2_vision_retrieval/` |
| `src/recovery/recovery_layer.py` | `src/track3_fusion_confidence/recovery_layer.py` |
| `src/fusion_confidence/fusion_engine.py` | `src/track3_fusion_confidence/fusion_engine.py` |
| `src/fusion_confidence/rejection.py` | `src/track3_fusion_confidence/rejection.py` |
| `tests/test_query_understanding/` | `tests/test_track1_query_understanding/` |
| `tests/test_vision_retrieval/` | `tests/test_track2_vision_retrieval/` |
| `tests/test_fusion_confidence/` | `tests/test_track3_fusion_confidence/` |

Git may show some empty `__init__.py` files as odd renames, such as `src/recovery/__init__.py` to `src/track2_vision_retrieval/indexing/__init__.py`. Git matches identical empty files, so these don't mean anything.

### Files deleted

| File | Why |
|---|---|
| `src/query_understanding/negation_verifier.py` | Unused stub; the Track 3 negation filter does this job, and the name clashed with the text check `verify_negation` in `track1_qwen.py` |
| `src/webapp/app.py` | Commented-out stub, replaced by `Web App/app.py` |

### Imports

These files changed only in their import paths:

- `scripts/benchmark_track2.py`
- `scripts/demo_track1_track2.py`
- `scripts/eval_calibration.py`
- `scripts/eval_equivalence.py`
- `scripts/eval_separation.py`
- `scripts/score_video.py`
- `scripts/test_faiss_real.py`
- `src/track2_vision_retrieval/retriever.py`
- All four test files in `tests/test_track2_vision_retrieval/`

`clip_encoder.py`, `two_pass_sampler.py`, `faiss_index.py` and `recovery_layer.py` are byte-for-byte unchanged.

## 2. Track 1: Query Understanding

### `decomposer.py`

`decompose_query()` was a stub that raised `NotImplementedError`. It now loads Qwen from the config, parses the query and returns a list of `SubQuery`.

### `track1_qwen.py`

`load_model()` now returns immediately if the model is already loaded, so the web demo doesn't reload it on every search. The prompt and parsing code are unchanged.

### `subquery_adapter.py`

These fixes came from real Qwen3-1.7B output:

| Problem | Example | Fix |
|---|---|---|
| Negation word stored as an attribute | `a without helmet`, kept as a positive | The word is removed and the entity counts as negated |
| A verb between the negation and the entity | `without wearing glasses` lost its negation | `-ing` verbs may now sit in between |
| Entity invented by Qwen | `a road`, which isn't in the query | Entities that aren't in the query and aren't used by any event are dropped |

## 3. Track 2: Vision & Retrieval

There are no logic changes. The default settings for `top_k_windows` (5) and `window_radius_seconds` (3.0) are also unchanged.

The pipeline now sends Track 2 only what must appear. Negated concepts are no longer sent to CLIP retrieval from `run_pipeline.py`. `demo_track1_track2.py` still passes them, because it displays their raw scores.

## 4. Track 3: Fusion, Confidence & Negation

The Track 3 modules came from the `fusion_confidence` folder added during this work.

| File | State |
|---|---|
| `fusion_engine.py` | Replaced the stub; one docstring updated for the new negation score |
| `rejection.py` | Replaced the stub, no other edits |
| `clip_exporter.py` | New, no edits |
| `query_adapter.py` | New; only a module path in a comment changed |
| `negation_filter.py` | New, and **its decision rule was changed** (below) |
| `recovery_layer.py` | Still a stub |

### Negation rule change

This is the most important behaviour change, and the Track 3 owner should review it.

| | Original rule | Current rule |
|---|---|---|
| Text sent to ConCLIP | The whole sentence, `a cat without a dog` | The excluded concept alone, `a dog` |
| A frame passes when | Similarity is **at or above** 0.20 | Similarity is **below** 0.205 |
| With several exclusions | Lowest similarity decides | Highest similarity decides |
| `negation_score` | Mean similarity | 1 minus the mean similarity |

Why it changed, from real tests on a video of four known scenes:

| Rule | Result |
|---|---|
| Original rule at its 0.20 cutoff | 5 of 12 cases correct; all 6 visible excluded things wrongly passed |
| Original rule at its best possible cutoff | 11 of 19 cases |
| Negated caption compared with affirmative caption | 16 of 19 cases at its best cutoff |
| Excluded concept alone (chosen) | 19 of 19 cases, which is also where its cutoff came from |
| Chosen rule on new concepts | 23 of 24 cases; it missed `smoke` in the rocket scene |

The original code comment cites 92% accuracy on NegBench COCO images. That result wasn't reproduced or disproved here, because this was a different test on one small video. A proper comparison on a dev set is still needed.

## 5. Pipeline: `scripts/run_pipeline.py`

The file was rewritten. It now has these parts:

| Function | Purpose |
|---|---|
| `parse_query` | Runs Track 1 and falls back to rules if Qwen fails or loses a negation |
| `build_segments` | **New logic:** turns frame scores into start and end times |
| `select_result` | Applies fusion, the threshold and the negation check, then ranks moments |
| `negation_status` | Reports whether ConCLIP can run |
| `load_negation_filter` | Loads ConCLIP once |
| `run` | Runs all three tracks, reports progress and returns a `PipelineReport` |

New command-line options: `--no-qwen`, `--skip-negation-check` and `--export`.

## 6. Web App (new)

| File | Purpose |
|---|---|
| `Web App/app.py` | FastAPI server: uploads, background searches, progress and status; runs on `127.0.0.1:8000` |
| `Web App/index.html` | The page: video drop zone and player, query box, result, track progress, score chart and table of candidate moments |
| `Web App/start_demo.bat` | Double-click launcher for Windows |

It runs one search at a time, deletes each uploaded video after its search, refuses files that aren't readable videos and limits uploads to 500 MB.

## 7. Configuration: `configs/config.yaml`

| Setting | Before | After | Note |
|---|---|---|---|
| `qwen3_model_name` | `Qwen/Qwen3-8B` (placeholder) | `Qwen/Qwen3-1.7B` | Matches the code |
| `confidence_threshold` | `0.28` | `0.185` | 0.28 rejected every real match; from one test video |
| `segment_score_drop` | none | `0.02` | New, uncalibrated |
| `negation.conclip_weights_path` | none | `data/conclip/conclip_b32_openclip_version.pt` | New |
| `negation.conclip_model_name` | none | `ViT-B-32` | New |
| `negation.frame_threshold` | none | `0.205` | New; from one test video |
| `negation.min_pass_ratio` | none | `0.7` | New; the Track 3 code's default |
| `negation.sample_interval_seconds` | none | `1.0` | New; the Track 3 code's default |
| `negation.max_frames_per_segment` | none | `16` | New; the Track 3 code's default |
| `negation.batch_size` | none | `16` | New; the Track 3 code's default |
| `negation.fusion_weight` | none | `0.5` | New, placeholder |

## 8. `requirements.txt`

- Added `open_clip_torch` and `pillow`.
- Removed a stray backtick on the last line, which could break `pip install`.

## 9. Tests

| File | Change |
|---|---|
| `tests/test_track1_query_understanding/test_subquery_adapter.py` | 2 new tests from real Qwen output; 1 test changed because `not wearing a hat` now counts as negated; 2 tests now use real query text |
| `tests/test_track2_vision_retrieval/` | Import paths only |
| `tests/test_track3_fusion_confidence/test_track3_modules.py` | New: Track 3 logic |
| `tests/test_track3_fusion_confidence/test_pipeline.py` | New: wiring across all tracks |
| `tests/test_web_app.py` | New: Web App requests |

## 10. Documentation

| File | Change |
|---|---|
| `README.md` | Updated structure, setup, demo, testing and current status |
| `docs/OPTIMIZATION_PLAN.md` | New |
| `docs/CHANGES_SINCE_CLONE.md` | New (this file) |

## 11. Optimization Round (13 Nov 2026)

| File | Change |
|---|---|
| `src/track2_vision_retrieval/embedding/frame_cache.py` | New: reuses frame embeddings per video (file hash), CLIP model and timestamp |
| `src/track2_vision_retrieval/scoring/softmax_confidence.py` | New: softmax of the query against the 80 COCO objects, leaving out the query itself and kinds of it |
| `src/track2_vision_retrieval/retriever.py` | Uses the frame cache; returns softmax `confidence` alongside raw `scores` |
| `src/common/types.py` | `FrameScores` gains an optional `confidence` field |
| `src/track1_query_understanding/track1_qwen.py` | Prompt prefix cache and `warm_up()`; output unchanged |
| `src/track1_query_understanding/decomposer.py` | Query cache and `warm_up()` |
| `scripts/run_pipeline.py` | Raw scores locate moments; softmax (or raw, by config) decides acceptance |
| `Web App/app.py`, `index.html` | Background model preloading, `--no-preload` and `--no-qwen-preload`, confidence chart |
| `configs/config.yaml` | `softmax_logit_scale: 100`, `confidence_score: "softmax"`, `confidence_threshold` 0.185 → 0.15 (softmax), raw value kept as `raw_confidence_threshold` |
| Tests | New `tests/conftest.py` and `test_softmax_and_cache.py`, plus retriever and pipeline tests; 82 in total |

Then, to stop the slow steps waiting for each other:

| File | Change |
|---|---|
| `src/track2_vision_retrieval/retriever.py` | New `prepare_video()`: embeds Pass 1 frames, and for videos up to `preindex_full_max_seconds` (60) the whole Pass 2 grid, before any query |
| `src/track2_vision_retrieval/embedding/frame_cache.py`, `clip_encoder.py` | Locks so preparation and a search on the same video never embed a frame twice or load CLIP twice |
| `scripts/run_pipeline.py` | With Qwen, Track 2 preparation runs while Track 1 parses; `--no-overlap` turns it off |
| `Web App/app.py`, `index.html` | Videos upload and start preparing when chosen (`POST /api/videos`); searches send the video id; the last 3 uploads are kept and deleted on exit |
| Tests | 90 in total |

Measured on the 20-second test video with a new query: 101s one step after the other, 78s overlapped,
47s with the video prepared on upload, 0.6s with Quick rules.

## Changes Outside the Repo

Teammates won't get these through git.

| Item | Detail |
|---|---|
| Python packages installed globally | `faiss-cpu` 1.15.1, `python-multipart` 0.0.33, `open_clip_torch` 4.2.1, plus its dependencies such as `ftfy` and `timm` |
| CLIP model | `openai/clip-vit-large-patch14`, 1.60 GB, in the Hugging Face cache |
| ConCLIP weights | `data/conclip/conclip_b32_openclip_version.pt`, 577 MB, ignored by git |

The folder `FinalYearProjectGIT/fusion_confidence` is the source the Track 3 files were copied from. It's outside the repo and no longer used.

## Not Done

- `recovery_layer.py` is still a stub, and `Web App` doesn't retry after a rejection.
- Thresholds still need proper calibration.
- Temporal order isn't enforced.
- FAISS isn't used by `retrieve()`.
- Speed improvements are planned but not made; see `OPTIMIZATION_PLAN.md`.
