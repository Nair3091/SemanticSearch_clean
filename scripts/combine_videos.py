"""
Combine several video clips into ONE clean video (re-encoded, so no broken joins).

Each clip is scaled to fit WIDTHxHEIGHT without stretching (black bars pad the
rest), set to a constant FPS, and encoded as H.264 with no audio. Phone-clip
rotation metadata is applied automatically by ffmpeg.

It prints the start/end time of each clip inside the combined video, which is the
ground truth you need for eval_separation.py / eval_calibration.py.

Run (from repo root):
  py scripts/combine_videos.py --inputs data/raw_videos/cat.mp4 data/raw_videos/dog.mp4 \
      --output data/raw_videos/cat_dog.mp4

Requires ffmpeg and ffprobe on PATH.
"""
import sys, argparse, shutil, subprocess
from pathlib import Path


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not read duration of {path}: {r.stderr.strip()}")


def decode_errors(path):
    """Fully decode the file; any stderr output means it is damaged."""
    r = run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    return r.stderr.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="clips in the order they should appear")
    p.add_argument("--output", default="data/raw_videos/combined.mp4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=23, help="H.264 quality (lower = better, 18 to 28 typical)")
    p.add_argument("--skip-check", action="store_true", help="skip the per-clip decode check")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"ERROR: {tool} not found on PATH")
    if len(a.inputs) < 2:
        sys.exit("ERROR: give at least two clips to combine")

    inputs = [Path(x) for x in a.inputs]
    for f in inputs:
        if not f.exists():
            sys.exit(f"ERROR: file not found: {f}")
    out = Path(a.output)
    if out.exists() and not a.overwrite:
        sys.exit(f"ERROR: {out} already exists (use --overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. check every source clip decodes cleanly before combining anything
    if not a.skip_check:
        bad = False
        for f in inputs:
            err = decode_errors(f)
            if err:
                bad = True
                print(f"DAMAGED: {f}\n  " + "\n  ".join(err.splitlines()[:3]))
            else:
                print(f"ok: {f}")
        if bad:
            sys.exit("\nERROR: fix or re-export the damaged clip(s) above, then re-run.")

    # 2. durations (used for the ground-truth ranges)
    durations = [get_duration(f) for f in inputs]

    # 3. build and run the ffmpeg command
    w, h = a.width, a.height
    parts = [
        f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={a.fps}[v{i}]"
        for i in range(len(inputs))
    ]
    concat = "".join(f"[v{i}]" for i in range(len(inputs))) + f"concat=n={len(inputs)}:v=1:a=0[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
    for f in inputs:
        cmd += ["-i", str(f)]
    cmd += ["-filter_complex", ";".join(parts + [concat]), "-map", "[v]",
            "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", str(out)]
    print("\nCombining...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit("ERROR: ffmpeg failed (see messages above)")

    # 4. verify the output
    err = decode_errors(out)
    total_out = get_duration(out)
    total_in = sum(durations)
    print(f"\nWrote {out}  ({total_out:.2f}s; sum of inputs {total_in:.2f}s)")
    if err:
        print("WARNING: the output has decode errors:\n  " + "\n  ".join(err.splitlines()[:3]))
    if abs(total_out - total_in) > 0.5:
        print("WARNING: output duration differs from the sum of the inputs by more than 0.5s")

    # 5. ground-truth time ranges inside the combined video
    print(f"\n{'clip':<28}{'start':>9}{'end':>9}{'duration':>10}")
    t = 0.0
    for f, d in zip(inputs, durations):
        print(f"{f.stem:<28}{t:>9.3f}{t + d:>9.3f}{d:>10.3f}")
        t += d
    print("\nUse these as --pos-start / --pos-end for a clip that contains the target.")
    print("Boundaries can drift by about 1 frame (0.03s) per clip; the eval scripts skip a 0.5s margin.")


if __name__ == "__main__":
    main()