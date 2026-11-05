"""
Compare each query's scores in the FIRST clip vs the SECOND clip of a combined video.

Reads the CSV written by score_video.py. Use it to check confusable siblings:
e.g. does "a cat" stay low in the dog clip, and "a dog" stay low in the cat clip?

Run (from repo root):
  py scripts/check_siblings.py --csv data/results/cat_dog_softmax_scores.csv --split 43.0

--split is the time (seconds) where the first clip ends, as printed by combine_videos.py.
Frames within --margin seconds of the split are ignored (boundary drift).
"""
import csv, argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--split", type=float, required=True, help="time (s) where the first clip ends")
p.add_argument("--margin", type=float, default=0.5)
a = p.parse_args()

rows = list(csv.reader(open(a.csv)))
queries = rows[0][1:]
d = np.array([[float(x) for x in r] for r in rows[1:]])
t, S = d[:, 0], d[:, 1:]

first = t < a.split - a.margin
second = t > a.split + a.margin
print(f"first clip: {first.sum()} frames (0 to {a.split - a.margin:.1f}s) | "
      f"second clip: {second.sum()} frames ({a.split + a.margin:.1f}s to {t[-1]:.1f}s)\n")

print(f"{'query':<14}{'mean first':>12}{'mean second':>13}{'max first':>11}{'max second':>12}")
for j, name in enumerate(queries):
    print(f"{name:<14}{S[first, j].mean():>12.3f}{S[second, j].mean():>13.3f}"
          f"{S[first, j].max():>11.3f}{S[second, j].max():>12.3f}")