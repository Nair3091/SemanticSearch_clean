from pathlib import Path

import cv2
import numpy as np

STEP, COLS, W = 2.0, 6, 320
out = Path("data/eval/sheets")
out.mkdir(parents=True, exist_ok=True)

for p in sorted(Path("data/raw_videos").glob("*.mp4")):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    tiles, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (W, int(f.shape[0] * W / f.shape[1])))
            cv2.putText(f, f"{t:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            tiles.append(f)
        t += STEP
    cap.release()
    if not tiles:
        continue
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(str(out / f"{p.stem}.png"), np.vstack(rows))
    print("wrote", out / f"{p.stem}.png")