import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from src.vision_retrieval.sampling.two_pass_sampler import coarse_sample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames

BACKGROUND = ["a photo", "an image", "a video frame", "a picture", "a scene"]
_COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
DISTRACTORS = [("an " if n[0] in "aeiou" else "a ") + n for n in _COCO]



def softmax_col0(logits):
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z[:, 0] / z.sum(axis=1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--pos-start", type=float, required=True)
    p.add_argument("--pos-end", type=float, required=True)
    p.add_argument("--present", nargs="*", default=["a dog", "a photo of a dog"])
    p.add_argument("--absent", nargs="*",
                   default=["a truck", "a spaceship", "a pizza", "a bicycle", "a boat"])
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.5)
    a = p.parse_args()

    frames = coarse_sample(a.video, a.interval)
    ts = np.array([f.timestamp for f in frames])
    inside = (ts >= a.pos_start + a.margin) & (ts <= a.pos_end - a.margin)
    outside = (ts < a.pos_start - a.margin) | (ts > a.pos_end + a.margin)
    keep = inside | outside

    F = embed_frames(frames)
    queries = a.present + a.absent
    is_present = np.array([True] * len(a.present) + [False] * len(a.absent))
    S = score_frames(F, embed_text(queries))                    # (N, Q)
    B = score_frames(F, embed_text(BACKGROUND)).mean(axis=1)    # (N,)
    D = score_frames(F, embed_text(DISTRACTORS))                # (N, K)

    sm = np.zeros_like(S)
    for j, q in enumerate(queries):
        mask = np.array([d.lower() != q.lower() for d in DISTRACTORS])
        logits = 100.0 * np.concatenate([S[:, j:j + 1], D[:, mask]], axis=1)
        sm[:, j] = softmax_col0(logits)

    methods = {"raw": S, "minus_bg": S - B[:, None], "softmax": sm}

    print(f"{len(ts)} frames | inside={inside.sum()} outside={outside.sum()} | "
          f"present={a.present} | absent={a.absent}")
    print(f"\n{'method':<10}{'pooled AUC':>12}{'peak margin':>13}{'margin/sd_neg':>15}")
    peaks, times = {}, {}
    for name, M in methods.items():
        pos, neg, pk, pt = [], [], [], []
        for j in range(len(queries)):
            s = M[:, j]
            if is_present[j]:
                pos.append(s[inside]); neg.append(s[outside]); pk.append(s[inside].max())
                pt.append(ts[inside][s[inside].argmax()])
            else:
                neg.append(s[keep]); pk.append(s[keep].max())
                pt.append(ts[keep][s[keep].argmax()])
        pos, neg, pk = np.concatenate(pos), np.concatenate(neg), np.array(pk)
        auc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        margin = pk[is_present].min() - pk[~is_present].max()
        print(f"{name:<10}{auc:>12.3f}{margin:>13.4f}{margin / neg.std():>15.2f}")
        peaks[name] = pk
        times[name] = np.array(pt)

    print("\nPer-query peak (present: inside range; absent: whole clip)")
    print(f"{'query':<20}{'present':>8}" + "".join(f"{n:>11}" for n in methods)
          + f"{'t@raw':>8}{'t@softmax':>11}")
    for j, q in enumerate(queries):
        print(f"{q:<20}{str(bool(is_present[j])):>8}"
              + "".join(f"{peaks[n][j]:>11.4f}" for n in methods)
              + f"{times['raw'][j]:>8.2f}{times['softmax'][j]:>11.2f}")


if __name__ == "__main__":
    main()