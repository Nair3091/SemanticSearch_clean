"""
Local demo for all three tracks. Upload a short video, type what to find, get the timestamp.

Start it from the repo root (it opens your browser):
    python "Web App/app.py"

Models load in the background at startup, so the first search does not pay for it. Options:
    --no-qwen-preload   do not preload Qwen (saves about 9 GB of RAM if you only use Quick rules)
    --no-preload        load every model on the first search instead
    --no-browser        do not open a browser tab

All pipeline logic lives in scripts/run_pipeline.py. This file only handles uploads, runs one
search at a time in the background, and reports progress to the page.

A video starts being prepared (Track 2 embeds its frames) as soon as it is uploaded, while you
type the query, and keeps going while Qwen parses it. Searches then only score frames.
"""

import atexit
import sys
import tempfile
import time
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from scripts import run_pipeline
from src.common.config import load_config

HOST, PORT = "127.0.0.1", 8000
PAGE = Path(__file__).resolve().parent / "index.html"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_QUERY_CHARS = 300
MAX_KEPT_VIDEOS = 3                           # older uploads are deleted

app = FastAPI(title="SemanticSearch demo")
_jobs = {}
_jobs_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # the models are shared, so searches run one at a time
_models = {"state": "not loaded"}             # "loading", "ready", or "failed: ..."
_preparer = ThreadPoolExecutor(max_workers=1) # embeds uploaded videos' frames, alongside searches
_videos = {}                                  # video_id -> path, duration, preparation state
_videos_lock = threading.Lock()


def warm_up(include_qwen: bool = True):
    """Load every model before the first search. Runs on the search worker, so a search started
    meanwhile simply waits instead of competing for the CPU."""
    _models["state"] = "loading"
    try:
        from src.track2_vision_retrieval import retriever
        from src.track2_vision_retrieval.embedding.clip_encoder import _load_model
        cfg = load_config()
        _load_model()
        retriever._bank_embeddings(cfg["vision_retrieval"]["clip_model_name"])
        if run_pipeline.negation_status(cfg)[0]:
            run_pipeline.load_negation_filter(cfg)
        if include_qwen:
            from src.track1_query_understanding import decomposer
            decomposer.warm_up()
        _models["state"] = "ready"
    except Exception as exc:   # a search will still try to load the models itself
        _models["state"] = f"failed: {type(exc).__name__}: {exc}"


def probe_video(path: str):
    """Duration in seconds, or None if OpenCV cannot read a frame from the file."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, _ = cap.read()
        return count / fps if ok and fps > 0 and count > 0 else None
    finally:
        cap.release()


def _update(job_id: str, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def report_to_dict(report, duration: float) -> dict:
    fs, parsed = report.frame_scores, report.parsed
    round_list = lambda values: [round(float(v), 4) for v in values]
    return {
        "duration": duration,
        "accepted": report.result.accepted,
        "message": report.result.message,
        "best": report.candidates[0] if report.candidates else None,
        "candidates": report.candidates,
        "threshold": report.threshold,
        "score_kind": report.score_kind,
        "parsed": parsed.to_dict(),
        "negation_checked": report.negation_checked,
        "timeline": {
            "timestamps": round_list(fs.timestamps),
            "fused": round_list(report.fused),
            "series": [{"label": q.text, "scores": round_list(s)} for q, s in zip(
                parsed.positive,
                fs.confidence if report.score_kind == "softmax" and fs.confidence is not None else fs.scores)],
            "windows": [[float(a), float(b)] for a, b in fs.windows],
        },
    }


def _run_job(job_id: str, video_path: str, query: str, use_qwen: bool, duration: float):
    def progress(track, info):
        with _jobs_lock:
            _jobs[job_id]["tracks"][track] = info

    try:
        ready, _ = run_pipeline.negation_status(load_config())
        # the upload already started preparing this video on _preparer, in parallel with this search
        report = run_pipeline.run(video_path, query, use_qwen=use_qwen, check_negation=ready,
                                  progress=progress, prepare_in_parallel=False)
        _update(job_id, state="done", result=report_to_dict(report, duration))
    except Exception as exc:   # shown on the page instead of leaving it waiting forever
        _update(job_id, state="error", error=f"{type(exc).__name__}: {exc}")


def _prepare(video_id: str, path: str):
    with _videos_lock:
        if video_id not in _videos:
            return
        _videos[video_id]["state"] = "preparing"
    t0 = time.time()
    try:
        info = run_pipeline.prepare_video(path)
        fields = {"state": "ready", "frames": info["frames"], "seconds": time.time() - t0}
    except Exception as exc:   # a search will embed what it needs itself
        fields = {"state": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - t0}
    with _videos_lock:
        if video_id in _videos:
            _videos[video_id].update(fields)


def _delete_all_videos():
    with _videos_lock:
        for v in _videos.values():
            Path(v["path"]).unlink(missing_ok=True)
        _videos.clear()


atexit.register(_delete_all_videos)


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/api/status")
def status():
    cfg = load_config()
    ready, reason = run_pipeline.negation_status(cfg)
    return {"negation_ready": ready, "negation_reason": reason, "models": _models["state"],
            "qwen_model": cfg["query_understanding"]["qwen3_model_name"],
            "clip_model": cfg["vision_retrieval"]["clip_model_name"],
            "devices": model_devices()}


def model_devices():
    """Where each loaded model runs ("cuda"/"gpu" or "cpu"); None until it has loaded."""
    import sys as _sys
    out = {"clip": None, "qwen": None, "conclip": None}
    clip = _sys.modules.get("src.track2_vision_retrieval.embedding.clip_encoder")
    if clip is not None and getattr(clip, "_model", None) is not None:
        out["clip"] = clip.device()
    qwen = _sys.modules.get("src.track1_query_understanding.track1_qwen")
    if qwen is not None and qwen._state.get("model") is not None:
        out["qwen"] = qwen._state.get("device")
    neg = getattr(run_pipeline, "_negation_filter", None)
    if neg is not None:
        out["conclip"] = getattr(neg, "device", None)
    return out


@app.post("/api/videos")
async def upload_video(video: UploadFile = File(...)):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        raise HTTPException(400, "Please upload a video file such as an MP4.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="semanticsearch_") as out:
        path, size = out.name, 0
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                Path(path).unlink(missing_ok=True)
                raise HTTPException(413, "That video is over the 500 MB limit.")
            out.write(chunk)

    duration = probe_video(path)
    if duration is None:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(400, "This file could not be read as a video.")

    video_id = uuid.uuid4().hex
    with _videos_lock:
        _videos[video_id] = {"path": path, "duration": duration, "state": "queued", "frames": 0,
                             "seconds": None, "error": None}
        while len(_videos) > MAX_KEPT_VIDEOS:
            oldest = next(iter(_videos))
            Path(_videos.pop(oldest)["path"]).unlink(missing_ok=True)
    _preparer.submit(_prepare, video_id, path)
    return {"video_id": video_id, "duration": duration}


@app.get("/api/videos/{video_id}")
def video_status(video_id: str):
    with _videos_lock:
        found = _videos.get(video_id)
        if found is None:
            raise HTTPException(404, "Unknown video. Please choose it again.")
        return {k: v for k, v in found.items() if k != "path"}


@app.post("/api/search")
def search(video_id: str = Form(...), query: str = Form(...), track1: str = Form("qwen")):
    query = query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise HTTPException(400, f"Enter a query of 1 to {MAX_QUERY_CHARS} characters.")
    if track1 not in ("qwen", "rules"):
        raise HTTPException(400, "Unknown Track 1 mode.")
    with _videos_lock:
        found = _videos.get(video_id)
        if found is None:
            raise HTTPException(404, "Unknown video. Please choose it again.")
        path, duration = found["path"], found["duration"]

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"state": "running", "query": query, "track1_mode": track1, "tracks": {},
                         "result": None, "error": None}
    _worker.submit(_run_job, job_id, path, query, track1 == "qwen", duration)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    with _jobs_lock:
        found = _jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "Unknown search.")
        return {**found, "tracks": dict(found["tracks"])}


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print(f"\nSemanticSearch demo: {url}   (press Ctrl+C to stop)\n")
    if "--no-preload" not in sys.argv:
        # --no-qwen-preload saves about 9 GB of RAM when you only use Quick rules
        _worker.submit(warm_up, "--no-qwen-preload" not in sys.argv)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
