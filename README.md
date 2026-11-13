# SemanticSearch
Negation-Aware Zero-Shot Text-to-Video Moment Retrieval Engine
22AIE498 — Team 1 — Guide: Aiswarya S Kumar

Describe a moment, including anything that must not be in it, and get its timestamp in a video.
For example: `a person riding a bike without a helmet`.

## Team & Ownership

| Track | Owner | Folder(s) |
|---|---|---|
| Track 1 — Query Understanding | Pulavarthi Vishnu Chaitanya | `src/track1_query_understanding/` |
| Track 2 — Vision & Retrieval | Dhruv Nair | `src/track2_vision_retrieval/` |
| Track 3 — Fusion, Confidence & Web Demo | Ihsal Riyas | `src/track3_fusion_confidence/`, `Web App/` |

## How It Works

```
query
  │  Track 1: Qwen3 + adapter, or quick rules
  ▼
what must appear + what must not appear
  │  Track 2: CLIP ViT-L/14 scores only what must appear
  │           Pass 1 samples every 2s, Pass 2 every 0.25s around the best frames
  ▼
frame scores over time
  │  Track 3: min-fusion, softmax confidence threshold, then the ConCLIP ViT-B/32 absence check
  ▼
best moment with start and end times, or a rejection
```

Negated concepts never go to CLIP retrieval, because CLIP would look for the thing the user wants
excluded. Instead, a frame passes the Track 3 check if its ConCLIP similarity to the excluded
concept is low.

Raw CLIP scores decide **where** a moment is. Whether to **accept** it is decided by a softmax
confidence: the query competes against the 80 COCO object names, and the score is the probability
that the query wins (0 to 1). Distractors that are the query itself or a kind of it are left out,
so `an animal` is not beaten by `a cat`. Set `confidence_score: "raw"` in the config to compare with
plain cosine scores.

## Project Structure

```
SemanticSearch_clean/
├── configs/
│   └── config.yaml            # ALL shared parameters live here — no hardcoding in modules
├── data/
│   ├── raw_videos/            # input video corpus (gitignored)
│   ├── frame_cache/           # extracted frames (gitignored)
│   ├── faiss_index/           # persisted FAISS index (gitignored)
│   ├── conclip/               # ConCLIP weights (gitignored, see Setup)
│   ├── benchmarks/            # saved Track 2 baseline
│   └── results/               # saved score plots and tables
├── docs/
│   ├── OPTIMIZATION_PLAN.md   # what to change to make all three tracks faster
│   └── CHANGES_SINCE_CLONE.md # everything changed since the first clone
├── scripts/
│   ├── run_pipeline.py        # THE end-to-end orchestrator — wires all 3 tracks together
│   ├── demo_track1_track2.py  # prints Track 1 and Track 2 output for one video
│   ├── benchmark_track2.py    # compares two-pass sampling with a full scan
│   └── eval_*.py              # Track 2 score and threshold experiments
├── src/
│   ├── common/
│   │   ├── config.py          # loads configs/config.yaml
│   │   └── types.py           # shared dataclasses (SubQuery, Frame, CandidateSegment, RetrievalResult)
│   │                          # import these everywhere — don't redefine locally
│   ├── track1_query_understanding/
│   │   ├── decomposer.py          # decompose_query(): the Track 1 entry point
│   │   ├── track1_qwen.py         # Qwen3 model, prompt and output checks
│   │   └── subquery_adapter.py    # Qwen JSON -> List[SubQuery]
│   ├── track2_vision_retrieval/
│   │   ├── retriever.py           # retrieve(): the Track 2 entry point
│   │   ├── sampling/two_pass_sampler.py
│   │   ├── embedding/clip_encoder.py
│   │   ├── embedding/frame_cache.py  # reuses frame embeddings across searches on a video
│   │   ├── scoring/softmax_confidence.py   # query vs 80 COCO distractors, 0 to 1
│   │   └── indexing/faiss_index.py   # not used by retrieve() yet
│   └── track3_fusion_confidence/
│       ├── query_adapter.py       # splits sub-queries into what must and must not appear
│       ├── fusion_engine.py       # min-fusion and the final score
│       ├── rejection.py           # confidence threshold
│       ├── negation_filter.py     # ConCLIP absence check
│       ├── clip_exporter.py       # cuts the result to an .mp4
│       └── recovery_layer.py      # not implemented yet
├── Web App/                   # local demo: upload a video, get the timestamp
│   ├── app.py
│   ├── index.html
│   └── start_demo.bat
├── tests/                     # mirrors src/ — one test folder per track
├── requirements.txt
└── README.md
```

## Ground Rules

1. **Shared types live in `src/common/types.py`.** If your function needs to hand data to
   someone else's track, use those dataclasses. Don't invent your own version of `SubQuery`
   or `Frame` inside your own folder.
2. **Config, not constants.** Any threshold, model name, or sampling interval goes in
   `configs/config.yaml`, not hardcoded inline. Everyone reads from the same file.
3. **`scripts/run_pipeline.py` is the only integration point.** Individual track modules
   should not import from each other directly — they get wired together in the
   orchestrator. This keeps each track independently testable.
4. **Tests mirror `src/`.** Before merging, add/update tests under the matching
   `tests/test_<track>/` folder.
5. **Branch naming:** `track1/<feature>`, `track2/<feature>`, `track3/<feature>` — merge to
   `main` via PR so the other two can review the interface (not necessarily the internals).

## Setup

Tested on Python 3.11. The commands below are for Windows; on Linux or macOS, activate with
`source .venv/bin/activate`.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt` lists `faiss-gpu`. Without a CUDA GPU, replace it with `faiss-cpu` before installing.

**Models.** Qwen3-1.7B (about 3.9 GB) and CLIP ViT-L/14 (about 1.6 GB) download automatically the
first time they are used.

**ConCLIP weights.** Negation checking also needs `conclip_b32_openclip_version.pt` (577 MB), which
is too large for git. Download it from the `models/ConCLIP` folder of the
[NegBench shared drive](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing)
and save it as `data/conclip/conclip_b32_openclip_version.pt`.

Without this file everything else still runs, and the demo clearly marks results as "negation not checked".

## Running the Web Demo

```bash
python "Web App/app.py"
```

This opens the demo at http://127.0.0.1:8000. On Windows you can also double-click
`Web App/start_demo.bat`.

The models load in the background when the app starts (about 1 to 2 minutes), so the first search
doesn't pay for it. Add `--no-qwen-preload` to skip loading Qwen (saves about 9 GB of RAM if you only
use Quick rules), or `--no-preload` to load everything on the first search instead.

1. Drop in a short video.
2. Describe what you are looking for.
3. Choose how to parse the query. **Quick rules** is instant and handles simple queries such as
   `a dog without a cat`. **Qwen3** runs the full Track 1 model and is much slower on a CPU.
4. Click **Find the moment** to get the time range, a button to play it, and each track's output.

## Running From the Command Line

```bash
python scripts/run_pipeline.py --video data/raw_videos/sample.mp4 --query "a person riding a bike without a helmet"
```

| Option | Effect |
|---|---|
| `--no-qwen` | Use rule-based query parsing |
| `--skip-negation-check` | Run without ConCLIP; results say the negation was not checked |
| `--export data/results/clip.mp4` | Save the best moment as a clip |

## Running the Tests

```bash
python -m pytest tests
```

The tests use stand-ins for Qwen, CLIP and ConCLIP, so they check the logic and wiring but not search
accuracy.

## Current Status

Tested on one 20-second video of four known scenes, on a CPU:

| Check | Result |
|---|---|
| Full searches (objects, phrases, general words, absent things, negation) | 18 of 19 correct at threshold 0.25; the 0.15 now used separates all 19 |
| Negation check on concepts not used to choose its cutoff | 23 of 24 |

| Speed | Before | Now |
|---|---|---|
| Track 2, first search on a video | ~38s | ~32s |
| Track 2, later searches on the same video | ~38s | 0.3–0.6s |
| Track 1 (Qwen), new query | 120–134s | 91–93s, identical output |
| Track 1 (Qwen), repeated query | 120–134s | instant |

These numbers come from one small video made from still photos, so they are not an accuracy measurement.

Known limitations:

- `confidence_threshold` (softmax) and `frame_threshold` come from that one video and still need
  calibrating. The softmax confidence only knows the 80 COCO objects as competitors, so things outside
  them compete less and general words outside the COCO groups may still score low.
  `segment_score_drop` and `fusion_weight` are uncalibrated placeholders.
- Qwen3-1.7B often misses negation. The adapter fixes several of its mistakes, and the pipeline falls
  back to rules if a negation word in the query was lost.
- `recovery_layer.py` is still a stub, so a low-confidence result is rejected without a retry.
- Temporal order such as "A then B" is extracted by Track 1 but not enforced.
- CLIP scores single frames, so it recognises objects better than actions.
- Things that aren't clear objects, such as smoke, are harder for the negation check.
- Long negated phrases can match other things in the frame, which may wrongly reject a moment.

Speed work is tracked in [docs/OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md).
