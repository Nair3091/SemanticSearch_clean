"""Demo: Track 1 (query understanding) -> Track 2 (vision retrieval) on one video.

  query --Qwen3--> entities/events/negation --adapter--> sub-queries --CLIP two-pass--> scores

Prints, for every sub-query, its best moments (distinct peaks) and a text timeline of the raw
CLIP scores. Negation inversion, fusion and accept/reject belong to Track 3 and are NOT done here,
so a NEGATED sub-query is scored as its positive text: a high score means that concept is present.

Examples (repo root, .venv active):
  py scripts/demo_track1_track2.py                                  # interactive loop
  py scripts/demo_track1_track2.py --query "a dog without a cat"
  py scripts/demo_track1_track2.py --query "a dog" --gt-start 17.125 --gt-end 35.27 --plot out.png
  py scripts/demo_track1_track2.py --pos "a dog" --neg "a cat"      # skip Track 1 (backup if Qwen misbehaves)
"""
import sys, argparse, csv, re, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.common.config import load_config
from src.common.types import SubQuery
from src.integration.subquery_adapter import structured_to_subqueries
from src.vision_retrieval.retriever import retrieve

BLOCKS = "▁▂▃▄▅▆▇█"
LABEL_W = 30


# ---------------------------------------------------------------- helpers
def video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps if fps > 0 else 0.0


def count_coarse(duration, interval):
    n, t = 0, 0.0
    while t < duration:
        n += 1
        t += interval
    return n


def top_peaks(ts, s, k, min_sep):
    """Indices of the k best frames that are at least min_sep seconds apart (adjacent frames
    are near-duplicates, so a plain top-k would just list one moment k times)."""
    picked = []
    for i in np.argsort(s)[::-1]:
        if all(abs(ts[i] - ts[j]) >= min_sep for j in picked):
            picked.append(int(i))
        if len(picked) == k:
            break
    return picked


def bucket_edges(duration, pass1_interval):
    cols = int(max(1, min(100, duration // pass1_interval)))
    return np.linspace(0.0, duration, cols + 1)


def sparkline(ts, s, edges):
    cols = len(edges) - 1
    vals = np.full(cols, np.nan)
    for c in range(cols):
        last = c == cols - 1
        m = (ts >= edges[c]) & ((ts <= edges[c + 1]) if last else (ts < edges[c + 1]))
        if m.any():
            vals[c] = s[m].max()
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    out = []
    for v in vals:
        if np.isnan(v):
            out.append("·")
        else:
            out.append(BLOCKS[0 if hi == lo else int(round((v - lo) / (hi - lo) * 7))])
    return "".join(out)


def numbered(path, n, loop):
    if not path or not loop:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{n}{p.suffix}"))


def label_of(q):
    return f"{q.text}{' (NEG)' if q.is_negated else ''}"


# ---------------------------------------------------------------- display
def show_track1(query, result, seconds, model_name):
    sq = result["structured_query"]
    print(f"\nTRACK 1  query understanding   ({model_name}, {seconds:.1f}s)")
    ents = []
    for e in sq.get("entities", []):
        attrs = f" [{', '.join(e['attributes'])}]" if e.get("attributes") else ""
        ents.append(f"{e['name']}{attrs} ({e['type']})")
    print("  entities :", " | ".join(ents) if ents else "(none)")
    events = [e for e in sq.get("events", []) if isinstance(e, dict)]
    if events:
        for ev in events:
            obj = ev.get("object")
            obj = f" -> {obj}" if obj not in (None, "", "null") else ""
            print(f"  event {ev.get('event_id')}  : {ev.get('subject')} --{ev.get('action')}{obj}"
                  f"    negated={ev.get('negated')}")
    else:
        print("  events   : (none)")
    rel = sq.get("temporal_relations", [])
    if rel:
        print("  temporal :", ", ".join(f"{r['event1']} before {r['event2']}" for r in rel),
              "(not used in scoring yet)")
    for v in result.get("negation_verification", []):
        if v.get("verification") == "REVIEW":
            real = [t for t in v.get("detected_triggers", [])
                    if re.search(rf"\b{re.escape(t)}\b", query.lower())]
            if real:   # ignore substring false alarms such as 'no' inside 'snow'
                print(f"  negation : REVIEW - '{', '.join(real)}' in the query but not tied to an event "
                      f"(adapter checks entities)")
        else:
            trig = f" (trigger '{v['negation_trigger']}')" if v.get("negation_trigger") else ""
            print(f"  negation : event {v.get('event_id')} {v.get('verification')}{trig}")


def show_subqueries(sub_queries, trace, note=None):
    print("\nADAPTER  sub-queries handed to Track 2")
    if note:
        print("  " + note)
    for line in trace:
        print("  " + line)


def show_track2(fs, sub_queries, args, c, duration, seconds):
    ts = fs.timestamps
    n_dense = int(fs.is_dense.sum())
    n_coarse = count_coarse(duration, c["pass1_interval_seconds"])
    full = int(np.ceil(duration / c["pass2_interval_seconds"]))
    covered = sum(max(0.0, min(e, duration) - s) for s, e in fs.windows)
    gt = (args.gt_start, args.gt_end) if args.gt_start is not None else None

    print(f"\nTRACK 2  vision retrieval   ({seconds:.1f}s)")
    print(f"  video {args.video} ({duration:.1f}s) | CLIP {c['clip_model_name']}")
    print(f"  pass 1: every {c['pass1_interval_seconds']}s over the whole video | pass 2: every "
          f"{c['pass2_interval_seconds']}s inside {len(fs.windows)} window(s) "
          f"{[(round(s, 1), round(e, 1)) for s, e in fs.windows]} = {100 * covered / duration:.0f}% of the video")
    print(f"  encoder calls: {n_coarse} + {n_dense} = {n_coarse + n_dense}  "
          f"(a full {c['pass2_interval_seconds']}s scan would need {full})")

    print("\nSCORES  raw CLIP cosine; higher = the frame matches the text better")
    for j, q in enumerate(sub_queries):
        s = fs.scores[j]
        tag = "NEGATED: a high score means the concept IS present" if q.is_negated else "positive"
        k = int(np.argmax(s))
        print(f"\n  [{j + 1}] \"{q.text}\"   ({tag})")
        print(f"      peak {s[k]:.4f} at {ts[k]:.2f}s | lowest {s.min():.4f}")
        head = f"      {'rank':<6}{'time(s)':>9}{'score':>9}{'pass':>7}" + ("   in ground truth" if gt else "")
        print(head)
        for r, i in enumerate(top_peaks(ts, s, args.top, args.min_sep), 1):
            inside = ""
            if gt:
                inside = "   yes" if gt[0] <= ts[i] <= gt[1] else "   no"
            print(f"      {r:<6}{ts[i]:>9.2f}{s[i]:>9.4f}{'dense' if fs.is_dense[i] else 'coarse':>7}{inside}")

    edges = bucket_edges(duration, c["pass1_interval_seconds"])
    print(f"\nTIMELINE  one column = {duration / (len(edges) - 1):.1f}s, bar = best score in the "
          f"column, scaled within each row (see range at the end)")
    for j, q in enumerate(sub_queries):
        s = fs.scores[j]
        print(f"  {label_of(q)[:LABEL_W - 1]:<{LABEL_W}}{sparkline(ts, s, edges)}  [{s.min():.3f}..{s.max():.3f}]")
    if gt:
        marks = "".join("^" if edges[c_] < gt[1] and edges[c_ + 1] > gt[0] else " "
                        for c_ in range(len(edges) - 1))
        print(f"  {'ground truth':<{LABEL_W}}{marks}")
    end = f"{duration:.0f}s"
    print(f"  {'':<{LABEL_W}}0s{' ' * max(1, len(edges) - 1 - 2 - len(end))}{end}")

    print("\nNote: negation inversion, fusion across sub-queries and accept/reject are Track 3 "
          "(not built); frames shown are the raw Track 2 output.")


def write_csv(path, fs, sub_queries):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "pass"] + [label_of(q) for q in sub_queries])
        for i, t in enumerate(fs.timestamps):
            w.writerow([f"{t:.3f}", "dense" if fs.is_dense[i] else "coarse"]
                       + [f"{fs.scores[j][i]:.5f}" for j in range(len(sub_queries))])
    print(f"  wrote {path}")


def make_plot(path, fs, sub_queries, gt, query):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("  (plot skipped: pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, q in enumerate(sub_queries):
        ax.plot(fs.timestamps, fs.scores[j], marker=".", ms=3, lw=1.2,
                linestyle="--" if q.is_negated else "-", label=label_of(q))
    handles, _ = ax.get_legend_handles_labels()
    for a, b in fs.windows:
        ax.axvspan(a, b, color="grey", alpha=0.12)
    handles.append(Patch(color="grey", alpha=0.25, label="pass-2 windows"))
    if gt:
        ax.axvspan(gt[0], gt[1], color="green", alpha=0.12)
        handles.append(Patch(color="green", alpha=0.25, label="ground truth"))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("raw CLIP cosine")
    ax.set_title(f"Track 1 + Track 2: \"{query}\"")
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- one query
def run_query(query, sub_queries, trace, args, c, max_sq, duration, n, loop, t1_info=None, note=None):
    from_track1 = t1_info is not None
    if from_track1:
        result, seconds, model_name = t1_info
        show_track1(query, result, seconds, model_name)
    show_subqueries(sub_queries, trace, note)
    t0 = time.time()
    fs = retrieve(args.video, sub_queries)
    show_track2(fs, sub_queries, args, c, duration, time.time() - t0)
    gt = (args.gt_start, args.gt_end) if args.gt_start is not None else None
    if args.csv:
        write_csv(numbered(args.csv, n, loop), fs, sub_queries)
    if args.plot:
        make_plot(numbered(args.plot, n, loop), fs, sub_queries, gt, query)


def main():
    p = argparse.ArgumentParser(description="Track 1 + Track 2 demo")
    p.add_argument("--video", default="data/raw_videos/eval_spliced_ar.mp4")
    p.add_argument("--query", help="run once with this query (omit for an interactive loop)")
    p.add_argument("--model", default=None, help="Qwen model name (default: the wrapper's default)")
    p.add_argument("--pos", nargs="*", default=[], help="skip Track 1: positive sub-queries")
    p.add_argument("--neg", nargs="*", default=[], help="skip Track 1: negated sub-queries")
    p.add_argument("--top", type=int, default=5, help="distinct peaks listed per sub-query")
    p.add_argument("--min-sep", type=float, default=2.0, help="min seconds between listed peaks")
    p.add_argument("--gt-start", type=float, help="optional ground-truth start (s)")
    p.add_argument("--gt-end", type=float, help="optional ground-truth end (s)")
    p.add_argument("--csv", help="write the per-frame score table here")
    p.add_argument("--plot", help="write a score-vs-time PNG here (needs matplotlib)")
    args = p.parse_args()
    if (args.gt_start is None) != (args.gt_end is None):
        p.error("give both --gt-start and --gt-end")
    if not Path(args.video).exists():
        p.error(f"video not found: {args.video}")

    cfg = load_config()
    c = cfg["vision_retrieval"]
    max_sq = cfg.get("query_understanding", {}).get("max_sub_queries", 5)
    duration = video_duration(args.video)

    manual = bool(args.pos or args.neg)
    t1 = None
    if not manual:
        from src.integration import track1_qwen as t1
        name = args.model or t1.DEFAULT_MODEL
        print(f"Loading Qwen ({name}) once ...")
        name, device = t1.load_model(name)
        print(f"  ready on {device}")
    print("Loading CLIP once ...")
    from src.vision_retrieval.embedding.clip_encoder import embed_text
    embed_text(["warm up"])
    print("  ready")

    if manual:
        sqs = [SubQuery(t, False) for t in args.pos] + [SubQuery(t, True) for t in args.neg]
        label = args.query or "manual sub-queries"
        trace = [f"manual -> \"{q.text}\" ({'NEGATED' if q.is_negated else 'positive'})" for q in sqs]
        run_query(label, sqs, trace, args, c, max_sq, duration, 1, False,
                  note="(Track 1 skipped: sub-queries given on the command line)")
        return

    def process(q, n, loop):
        try:
            t0 = time.time()
            result = t1.understand_query(q)
            info = (result, time.time() - t0, name)
            sqs, trace = structured_to_subqueries(result, q, max_sq)
            note = None
        except (ValueError, RuntimeError) as exc:
            print(f"\nTRACK 1 failed: {str(exc)[:300]}")
            info, note = None, "Track 1 failed, falling back to the raw query as one positive sub-query"
            sqs, trace = [SubQuery(q.strip(), False)], [f"fallback -> \"{q.strip()}\" (positive)"]
        run_query(q, sqs, trace, args, c, max_sq, duration, n, loop, info, note)

    if args.query:
        process(args.query.strip(), 1, False)
    else:
        n = 0
        while True:
            q = input("\nQuery (blank to quit): ").strip()
            if not q:
                break
            n += 1
            process(q, n, True)


if __name__ == "__main__":
    main()
