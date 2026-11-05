"""
Score a WHOLE video against any list of queries (no ground-truth timestamps).

For every sampled frame it computes the CLIP score for each query, then prints
per-query max / time of max / mean / top distinct peaks, saves a CSV of all
scores, and saves a score-vs-time plot.

Run (from repo root):
  py scripts/score_video.py --video data/raw_videos/eval_spliced_ar.mp4 \
      --queries "a dog" "an animal" "a pet" "a cat" "a horse" "a pizza" "a spaceship" "a truck"

Optional: --softmax scores each query relative to an 80-class COCO bank
(logit scale 100, the query itself masked out) instead of raw cosine.
"""
import sys, argparse, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.vision_retrieval.sampling.two_pass_sampler import coarse_sample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames

_COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
         "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
         "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
         "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
         "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
         "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
         "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
         "toothbrush"]
BANK = [("an " if n[0] in "aeiou" else "a ") + n for n in _COCO]


def softmax_scores(S, F, queries):
    D = score_frames(F, embed_text(BANK))
    out = np.zeros_like(S)
    for j, q in enumerate(queries):
        mask = np.array([d.lower() != q.lower() for d in BANK])
        logits = 100.0 * np.concatenate([S[:, j:j + 1], D[:, mask]], axis=1)
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        out[:, j] = z[:, 0] / z.sum(axis=1)
    return out


def top_peaks(ts, s, k, min_sep):
    """Highest-scoring frames that are at least min_sep seconds apart."""
    peaks = []
    for i in np.argsort(s)[::-1]:
        if all(abs(ts[i] - ts[j]) >= min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) == k:
            break
    return peaks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--queries", nargs="+", required=True)
    p.add_argument("--interval", type=float, default=0.5, help="seconds between sampled frames")
    p.add_argument("--peaks", type=int, default=3, help="distinct peaks to list per query")
    p.add_argument("--min-sep", type=float, default=2.0, help="min seconds between listed peaks")
    p.add_argument("--softmax", action="store_true", help="relative score vs COCO bank instead of raw cosine")
    p.add_argument("--out-dir", default="data/results")
    a = p.parse_args()

    frames = coarse_sample(a.video, a.interval)
    ts = np.array([f.timestamp for f in frames])
    F = embed_frames(frames)
    S = score_frames(F, embed_text(a.queries))             # (N, Q) raw cosine
    M = softmax_scores(S, F, a.queries) if a.softmax else S
    kind = "softmax" if a.softmax else "raw cosine"

    print(f"{len(ts)} frames, every {a.interval}s, 0.0s to {ts[-1]:.1f}s | score = {kind}\n")
    print(f"{'query':<22}{'max':>8}{'t_max':>8}{'mean':>8}   top peaks (time:score)")
    for j, q in enumerate(a.queries):
        s = M[:, j]
        pk = top_peaks(ts, s, a.peaks, a.min_sep)
        peaks_txt = "  ".join(f"{ts[i]:.1f}:{s[i]:.4f}" for i in pk)
        print(f"{q:<22}{s.max():>8.4f}{ts[int(np.argmax(s))]:>8.1f}{s.mean():>8.4f}   {peaks_txt}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem + ("_softmax" if a.softmax else "_raw")

    csv_path = out / f"{stem}_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + a.queries)
        for i in range(len(ts)):
            w.writerow([f"{ts[i]:.3f}"] + [f"{M[i, j]:.5f}" for j in range(len(a.queries))])
    print(f"\nSaved per-frame scores: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        for j, q in enumerate(a.queries):
            ax.plot(ts, M[:, j], label=q, linewidth=1.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(kind)
        ax.set_title(Path(a.video).name)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        png_path = out / f"{stem}_plot.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=130)
        print(f"Saved plot: {png_path}")
    except ImportError:
        print("matplotlib not installed; skipped plot (pip install matplotlib)")


if __name__ == "__main__":
    main()