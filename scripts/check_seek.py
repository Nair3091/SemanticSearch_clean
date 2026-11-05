import sys
import cv2, numpy as np

path = sys.argv[1]
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
targets = sorted(set(int(t * fps) for t in np.arange(0, n / fps, 0.5)))
tset, seq, i = set(targets), {}, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i in tset:
        seq[i] = f
    i += 1
cap.release()

cap = cv2.VideoCapture(path)
for idx in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    d = np.abs(f.astype(int) - seq[idx].astype(int)).mean() if ok and idx in seq else float("nan")
    print(idx, round(float(d), 3))