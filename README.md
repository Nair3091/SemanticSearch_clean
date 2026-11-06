# SemanticSearch
Negation-Aware Zero-Shot Text-to-Video Moment Retrieval Engine
22AIE498 — Team 1 — Guide: Aiswarya S Kumar

## Team & Ownership

| Track | Owner | Folder(s) |
|---|---|---|
| Track 1 — Query Understanding | Pulavarthi Vishnu Chaitanya | `src/track1_query_understanding/` |
| Track 2 — Vision & Retrieval | Dhruv Nair | `src/track2_vision_retrieval/` |
| Track 3 — Fusion, Confidence & Web Demo | Ihsal Riyas | `src/track3_fusion_confidence/`, `Web App/` |

## Project Structure

```
SemanticSearch/
├── configs/
│   └── config.yaml            # ALL shared parameters live here — no hardcoding in modules
├── data/
│   ├── raw_videos/            # input video corpus (gitignored)
│   ├── frame_cache/           # extracted frames (gitignored)
│   └── faiss_index/           # persisted FAISS index (gitignored)
├── scripts/
│   └── run_pipeline.py        # THE end-to-end orchestrator — wires all 3 tracks together
├── src/
│   ├── common/
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
│   │   └── indexing/faiss_index.py
│   └── track3_fusion_confidence/
│       ├── query_adapter.py       # splits sub-queries into CLIP queries and negation captions
│       ├── fusion_engine.py       # min-fusion and the final score
│       ├── rejection.py           # confidence threshold
│       ├── negation_filter.py     # ConCLIP negation check
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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Negation checking also needs the ConCLIP weights. Download `conclip_b32_openclip_version.pt` from the
[NegBench models folder](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing)
and save it as `data/conclip/conclip_b32_openclip_version.pt`.

## Running the web demo

```bash
python "Web App/app.py"
```

This opens the demo at http://127.0.0.1:8000. On Windows you can also double-click
`Web App/start_demo.bat`. On a CPU, expect about 1.5 minutes of Track 2 for a 20-second video, plus
2 to 5 minutes of Track 1 if you choose the Qwen3 option.

## Running the pipeline from the command line

```bash
python scripts/run_pipeline.py --video data/raw_videos/sample.mp4 --query "a person riding a bike without a helmet"
```

Add `--no-qwen` for rule-based query parsing, `--skip-negation-check` to run without ConCLIP, or
`--export data/results/clip.mp4` to save the best moment as a clip.
