"""Does the two-pass retriever find the same peak as a full uniform scan?

Full scan  = every frame on the Pass-2 grid (pass2_interval_seconds) over the whole video.
Two-pass   = retrieve(), exactly as the pipeline calls it (config-driven).

Both are scored with the same CLIP model, so on shared timestamps the scores must match
(sanity check 1). The real question is whether two-pass reaches the full-scan maximum
(check 2). Run from the repo root with .venv active.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.common.config import load_config
from src.common.types import SubQuery
from src.vision_retrieval.sampling.two_pass_sampler import dense_resample
from src.vision_retrieval.embedding.clip_encoder import embed_frames, embed_text, score_frames
from src.vision_retrieval.retriever import retrieve

TOL = 1e-6


def video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps if fps > 0 else 0.0


def full_scan(video, interval, text_emb, duration, chunk_seconds=20.0):
    """Score every frame on a uniform grid, in chunks so RAM stays small."""
    ts, rows = [], []
    start = 0.0
    while start < duration:
        frames = dense_resample(video, start, start + chunk_seconds - interval, interval)
        if frames:
            ts.extend(f.timestamp for f in frames)
            rows.append(score_frames(embed_frames(frames), text_emb))
        start += chunk_seconds
    return np.array(ts), np.concatenate(rows, axis=0)


def guide_curve(S, negated):
    """Same rule as retrieve(): min over positive sub-queries, else -max over all."""
    pos = [j for j, n in enumerate(negated) if not n]
    return S[:, pos].min(axis=1) if pos else -S.max(axis=1)


def count_coarse(duration, interval):
    n, t = 0, 0.0
    while t < duration:
        n += 1
        t += interval
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--positive", nargs="*", default=[])
    p.add_argument("--negated", nargs="*", default=[])
    a = p.parse_args()
    sq = [SubQuery(t, False) for t in a.positive] + [SubQuery(t, True) for t in a.negated]
    if not sq:
        p.error("give at least one of --positive / --negated")
    negated = [q.is_negated for q in sq]

    c = load_config()["vision_retrieval"]
    interval = c["pass2_interval_seconds"]
    duration = video_duration(a.video)

    tp = retrieve(a.video, sq)
    text_emb = embed_text([q.text for q in sq])
    t_full, S_full = full_scan(a.video, interval, text_emb, duration)
    S_tp = np.stack(tp.scores, axis=1)

    # ---- sanity check 1: same frame, same score ----
    idx = {round(float(t), 3): i for i, t in enumerate(t_full)}
    tp_keys = [round(float(t), 3) for t in tp.timestamps]
    shared_tp = [i for i, k in enumerate(tp_keys) if k in idx]
    shared_full = [idx[tp_keys[i]] for i in shared_tp]
    off_grid = len(tp_keys) - len(shared_tp)
    max_diff = float(np.abs(S_tp[shared_tp] - S_full[shared_full]).max()) if shared_tp else float("nan")

    n_coarse = count_coarse(duration, c["pass1_interval_seconds"])
    n_dense = int(tp.is_dense.sum())
    calls_tp = n_coarse + n_dense
    covered = sum(max(0.0, min(e, duration) - s) for s, e in tp.windows)

    print(f"video {duration:.2f}s | pass1 {c['pass1_interval_seconds']}s, pass2 {interval}s, "
          f"top_k {c['top_k_windows']}, radius {c['window_radius_seconds']}s")
    print(f"windows {[(round(s, 2), round(e, 2)) for s, e in tp.windows]} "
          f"cover {covered:.1f}s = {100 * covered / duration:.0f}% of the video")
    print(f"encoder calls: two-pass {n_coarse} + {n_dense} = {calls_tp} | "
          f"full scan {len(t_full)} | ratio {len(t_full) / calls_tp:.2f}x fewer")
    print(f"check 1 (same frame, same score): {len(shared_tp)} shared timestamps, "
          f"{off_grid} off-grid, max |diff| = {max_diff:.2e}  "
          f"{'OK' if max_diff < 1e-4 else 'MISMATCH'}")

    # ---- check 2: does two-pass reach the full-scan maximum? ----
    rows = [(f"{q.text}{' (neg)' if q.is_negated else ''}", S_full[:, j], S_tp[:, j])
            for j, q in enumerate(sq)]
    label = "guide min(pos)" if not all(negated) else "guide -max"
    rows.append((label, guide_curve(S_full, negated), guide_curve(S_tp, negated)))

    print(f"\n{'curve':<22}{'full t':>8}{'full peak':>11}{'2-pass t':>10}{'2-pass peak':>13}"
          f"{'gap':>9}{'rank in full':>14}")
    hits = 0
    for name, cf, ct in rows:
        i_f, j_t = int(cf.argmax()), int(ct.argmax())
        gap = cf[i_f] - ct[j_t]
        rank = 1 + int((cf > ct[j_t] + TOL).sum())    # 1 = two-pass best is a global max
        hits += rank == 1
        print(f"{name:<22}{t_full[i_f]:>8.2f}{cf[i_f]:>11.4f}{tp.timestamps[j_t]:>10.2f}"
              f"{ct[j_t]:>13.4f}{gap:>9.4f}{rank:>14d}")
    print(f"\ntwo-pass reached the full-scan maximum on {hits}/{len(rows)} curves "
          f"(rank 1 = yes; gap is full peak minus two-pass peak)")


if __name__ == "__main__":
    main()
