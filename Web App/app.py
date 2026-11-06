"""
Local demo for all three tracks. Upload a short video, type what to find, get the timestamp.

Start it from the repo root (it opens your browser):
    python "Web App/app.py"

All pipeline logic lives in scripts/run_pipeline.py. This file only handles uploads, runs one
search at a time in the background, and reports progress to the page.
"""

import sys
import tempfile
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

app = FastAPI(title="SemanticSearch demo")
_jobs = {}
_jobs_lock = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1)   # the models are shared, so searches run one at a time


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
        "threshold": load_config()["fusion_confidence"]["confidence_threshold"],
        "parsed": parsed.to_dict(),
        "negation_checked": report.negation_checked,
        "timeline": {
            "timestamps": round_list(fs.timestamps),
            "fused": round_list(report.fused),
            "series": [{"label": q.text, "scores": round_list(s)} for q, s in zip(parsed.positive, fs.scores)],
            "windows": [[float(a), float(b)] for a, b in fs.windows],
        },
    }


def _run_job(job_id: str, video_path: str, query: str, use_qwen: bool, duration: float):
    def progress(track, info):
        with _jobs_lock:
            _jobs[job_id]["tracks"][track] = info

    try:
        ready, _ = run_pipeline.negation_status(load_config())
        report = run_pipeline.run(video_path, query, use_qwen=use_qwen, check_negation=ready, progress=progress)
        _update(job_id, state="done", result=report_to_dict(report, duration))
    except Exception as exc:   # shown on the page instead of leaving it waiting forever
        _update(job_id, state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        Path(video_path).unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/api/status")
def status():
    cfg = load_config()
    ready, reason = run_pipeline.negation_status(cfg)
    return {"negation_ready": ready, "negation_reason": reason,
            "qwen_model": cfg["query_understanding"]["qwen3_model_name"],
            "clip_model": cfg["vision_retrieval"]["clip_model_name"]}


@app.post("/api/search")
async def search(video: UploadFile = File(...), query: str = Form(...), track1: str = Form("qwen")):
    query = query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise HTTPException(400, f"Enter a query of 1 to {MAX_QUERY_CHARS} characters.")
    if track1 not in ("qwen", "rules"):
        raise HTTPException(400, "Unknown Track 1 mode.")

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
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
