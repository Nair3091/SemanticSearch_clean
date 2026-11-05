import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from src.common.config import load_config
from src.vision_retrieval.sampling.two_pass_sampler import coarse_sample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--pos-start", type=float, required=True)
    p.add_argument("--pos-end", type=float, required=True)
    p.add_argument("--target", default="a dog")
    p.add_argument("--contrast", nargs="*", default=["a cat", "an empty room"])
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--threshold", type=float,
                   default=load_config()["fusion_confidence"]["confidence_threshold"])
    a = p.parse_args()

    frames = coarse_sample(a.video, a.interval)
    ts = np.array([f.timestamp for f in frames])
    inside = (ts >= a.pos_start + a.margin) & (ts <= a.pos_end - a.margin)
    outside = (ts < a.pos_start - a.margin) | (ts > a.pos_end + a.margin)
    keep = inside | outside
    in_range = (ts >= a.pos_start) & (ts <= a.pos_end)
    print(f"{len(ts)} frames | inside={inside.sum()} outside={outside.sum()} dropped={(~keep).sum()}")

    queries = [a.target] + a.contrast
    S = score_frames(embed_frames(frames), embed_text(queries))

    print(f"\n{'query':<22}{'mean_in':>9}{'mean_out':>10}{'gap':>8}{'d':>7}{'AUC':>7}  top1_t  in_range")
    for j, q in enumerate(queries):
        s = S[:, j]
        x, y = s[inside], s[outside]
        pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        d = (x.mean() - y.mean()) / pooled          # effect size (Cohen's d)
        auc = roc_auc_score(inside[keep], s[keep])
        top = int(np.argmax(s))
        print(f"{q:<22}{x.mean():>9.4f}{y.mean():>10.4f}{x.mean() - y.mean():>8.4f}{d:>7.2f}{auc:>7.3f}  {ts[top]:>5.1f}  {bool(in_range[top])}")

    t = S[:, 0]
    print(f"\nTarget '{a.target}' @ threshold {a.threshold}:")
    print(f"  recall (inside >= thr):       {(t[inside] >= a.threshold).mean():.2f}")
    print(f"  false-pos (outside >= thr):   {(t[outside] >= a.threshold).mean():.2f}")
    print(f"  min inside {t[inside].min():.4f} | max outside {t[outside].max():.4f} | "
          f"overlap gap {t[inside].min() - t[outside].max():+.4f}")
    if S.shape[1] > 1:
        best = S[:, 1:].max(axis=1)
        m = t[inside] - best[inside]
        print(f"  beats every contrast query on {(m > 0).mean():.0%} of inside frames | mean margin {m.mean():+.4f}")


if __name__ == "__main__":
    main()