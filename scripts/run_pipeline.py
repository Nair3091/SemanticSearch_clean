"""
End-to-end pipeline orchestrator — this is the ONE place that wires
Track 1 -> Track 2 -> Track 3 together.

Nobody should call another track's internals directly from inside their
own track's modules except through the function signatures defined here
and in src/common/types.py. Keeps the three tracks independently testable.

Run: python scripts/run_pipeline.py --video path/to/video.mp4 --query "..."
"""

import argparse
from src.vision_retrieval.retriever import retrieve
from src.query_understanding.decomposer import decompose_query
from src.vision_retrieval.sampling.two_pass_sampler import coarse_sample, dense_resample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames
from src.vision_retrieval.indexing.faiss_index import load_index, search
from src.fusion_confidence.fusion_engine import fuse_scores
from src.fusion_confidence.rejection import check_confidence
from src.recovery.recovery_layer import relax_query, widen_window


def run(video_path: str, query: str) -> None:
    # 1. Query Understanding (Track 1)
    sub_queries = decompose_query(query)

        # 2. Vision & Retrieval (Track 2)
    frame_scores = retrieve(video_path, sub_queries)
    # frame_scores.scores[j] aligns with sub_queries[j]; Track 3 handles negation, fusion, threshold

    raise NotImplementedError("Wire up the full pipeline as each track's modules land")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    run(args.video, args.query)
