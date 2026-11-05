import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.common.config import load_config
from src.vision_retrieval.sampling.two_pass_sampler import coarse_sample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames
from src.vision_retrieval.indexing.faiss_index import build_index, load_index, search

VIDEO = "data/raw_videos/eval_spliced.mp4"
path = load_config()["vision_retrieval"]["faiss_index_path"]

frames = coarse_sample(VIDEO, 0.5)
fe = embed_frames(frames)
build_index(fe, path)
index = load_index(path)

te = embed_text(["a photo of a dog"])
scores, idx = search(index, te, top_k=5)
brute = score_frames(fe, te)[:, 0]
brute_idx = np.argsort(brute)[::-1][:5]

print("faiss :", [(frames[i].timestamp, round(float(s), 4)) for i, s in zip(idx, scores)])
print("brute :", [(frames[i].timestamp, round(float(brute[i]), 4)) for i in brute_idx])
print("match:", list(idx) == list(brute_idx))