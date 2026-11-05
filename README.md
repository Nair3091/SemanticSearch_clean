# SemanticSearch
Negation-Aware Zero-Shot Text-to-Video Moment Retrieval Engine
22AIE498 — Team 1 — Guide: Aiswarya S Kumar

## Team & Ownership

| Track | Owner | Folder(s) |
|---|---|---|
| Track 1 — Query Understanding | Pulavarthi Vishnu Chaitanya | `src/query_understanding/` |
| Track 2 — Vision & Retrieval | Dhruv Nair | `src/vision_retrieval/` |
| Track 3 — Fusion, Confidence & Web Demo | Ihsal Riyas | `src/fusion_confidence/`, `src/recovery/`, `src/webapp/` |

## Project Structure

```
SemanticSearch/
├── configs/
│   └── config.yaml            # ALL shared parameters live here — no hardcoding in modules
├── data/
│   ├── raw_videos/            # input video corpus (gitignored)
│   ├── frame_cache/           # extracted frames (gitignored)
│   └── faiss_index/           # persisted FAISS index (gitignored)
├── docs/                      # design doc, diagrams, report drafts
├── notebooks/                 # exploratory work, each person prefixes with their initials
├── scripts/
│   └── run_pipeline.py        # THE end-to-end orchestrator — wires all 3 tracks together
├── src/
│   ├── common/
│   │   └── types.py           # shared dataclasses (SubQuery, Frame, CandidateSegment, RetrievalResult)
│   │                          # import these everywhere — don't redefine locally
│   ├── query_understanding/   # Track 1
│   │   ├── decomposer.py
│   │   └── negation_verifier.py
│   ├── vision_retrieval/      # Track 2
│   │   ├── sampling/two_pass_sampler.py
│   │   ├── embedding/clip_encoder.py
│   │   └── indexing/faiss_index.py
│   ├── fusion_confidence/     # Track 3
│   │   ├── fusion_engine.py
│   │   └── rejection.py
│   ├── recovery/              # Track 3
│   │   └── recovery_layer.py
│   └── webapp/                # Track 3
│       └── app.py
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

## Running the pipeline (once modules are implemented)

```bash
python scripts/run_pipeline.py --video data/raw_videos/sample.mp4 --query "a person riding a bike without a helmet"
```
