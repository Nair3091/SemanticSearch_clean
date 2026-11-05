import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw_videos")
print(f"{'file':<28}{'WxH':>11}{'shape':>11}{'fps':>8}{'frames':>8}{'sec':>8}")
for p in sorted(root.glob("*.mp4")):
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"{p.name:<28} could not open")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    shape = "portrait" if h > w else ("landscape" if w > h else "square")
    sec = n / fps if fps > 0 else 0.0
    print(f"{p.name:<28}{f'{w}x{h}':>11}{shape:>11}{fps:>8.2f}{n:>8}{sec:>8.1f}")