"""Web App request handling. The pipeline is replaced by a stand-in; real runs are not covered here."""
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.common.types import CandidateSegment, FrameScores, RetrievalResult, SubQuery
from src.track3_fusion_confidence.query_adapter import from_sub_queries

APP_PATH = Path(__file__).resolve().parent.parent / "Web App" / "app.py"
spec = importlib.util.spec_from_file_location("semanticsearch_web_app", APP_PATH)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)
client = TestClient(web.app)


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    w.release()
    return path.read_bytes()


def fake_report(query):
    parsed = from_sub_queries(query, [SubQuery("a dog", False)])
    ts = np.array([0.0, 1.0, 2.0])
    frame_scores = FrameScores(ts, [np.array([0.1, 0.4, 0.2])], np.ones(3, dtype=bool), [(0.0, 2.0)])
    candidate = {"start": 1.0, "end": 1.25, "retrieval_confidence": 0.4, "final_score": 0.4,
                 "status": "accepted", "negation_score": None, "pass_ratio": None, "frames_checked": 0}
    return web.run_pipeline.PipelineReport(
        parsed, frame_scores, RetrievalResult(True, CandidateSegment(1.0, 1.25, 0.4), "Accepted"),
        [candidate], np.array([0.1, 0.4, 0.2]), False, {})


def wait_for(job_id):
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    pytest.fail("job did not finish")


@pytest.fixture(autouse=True)
def fake_prepare(monkeypatch):
    """Uploads start Track 2 preparation; replace it so no model loads."""
    calls = []

    def prepare(path):
        calls.append(path)
        return {"frames": 12, "duration": 3.0, "full_grid": True}

    monkeypatch.setattr(web.run_pipeline, "prepare_video", prepare)
    return calls


def upload(video_bytes, name="v.avi"):
    return client.post("/api/videos", files={"video": (name, video_bytes, "video/avi")})


def wait_ready(video_id):
    for _ in range(100):
        v = client.get(f"/api/videos/{video_id}").json()
        if v["state"] in ("ready", "failed"):
            return v
        time.sleep(0.05)
    pytest.fail("video was not prepared")


def search(video_id, query="a dog", track1="rules"):
    return client.post("/api/search", data={"video_id": video_id, "query": query, "track1": track1})


def test_page_and_status_load():
    assert "SemanticSearch" in client.get("/").text
    assert {"negation_ready", "negation_reason", "models"} <= set(client.get("/api/status").json())


def test_upload_starts_preparing_the_video(video_bytes, fake_prepare):
    body = upload(video_bytes).json()
    assert body["duration"] == pytest.approx(3.0)
    v = wait_ready(body["video_id"])
    assert v["state"] == "ready" and v["frames"] == 12 and "path" not in v
    assert len(fake_prepare) == 1 and Path(fake_prepare[0]).exists()


def test_search_returns_timestamp_and_keeps_the_video_for_more_searches(video_bytes, monkeypatch):
    seen = []

    def fake_run(path, query, use_qwen, check_negation, progress, prepare_in_parallel):
        seen.append(dict(path=path, query=query, use_qwen=use_qwen, overlap=prepare_in_parallel))
        progress("track1", {"state": "done", "seconds": 0.1})
        return fake_report(query)

    monkeypatch.setattr(web.run_pipeline, "run", fake_run)
    video_id = upload(video_bytes).json()["video_id"]
    for query in ["a dog", "a cat"]:
        job = wait_for(search(video_id, query).json()["job_id"])
        assert job["state"] == "done", job["error"]
    assert job["result"]["accepted"] and job["result"]["best"]["start"] == 1.0
    assert job["result"]["duration"] == pytest.approx(3.0)
    assert job["tracks"]["track1"]["state"] == "done"
    assert [s["query"] for s in seen] == ["a dog", "a cat"] and seen[0]["use_qwen"] is False
    assert seen[0]["overlap"] is False            # the upload already prepared the video
    assert seen[0]["path"] == seen[1]["path"] and Path(seen[0]["path"]).exists()


def test_old_uploads_are_deleted(video_bytes, monkeypatch):
    monkeypatch.setattr(web, "MAX_KEPT_VIDEOS", 1)
    first = upload(video_bytes).json()["video_id"]
    wait_ready(first)
    path = web._videos[first]["path"]
    upload(video_bytes)
    assert first not in web._videos and not Path(path).exists()
    assert client.get(f"/api/videos/{first}").status_code == 404


def test_pipeline_error_reaches_the_page(video_bytes, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("ConCLIP weights not found")

    monkeypatch.setattr(web.run_pipeline, "run", boom)
    job = wait_for(search(upload(video_bytes).json()["video_id"]).json()["job_id"])
    assert job["state"] == "error" and "ConCLIP weights" in job["error"]


def test_failed_preparation_is_reported(video_bytes, monkeypatch):
    def boom(path):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(web.run_pipeline, "prepare_video", boom)
    v = wait_ready(upload(video_bytes).json()["video_id"])
    assert v["state"] == "failed" and "out of memory" in v["error"]


@pytest.mark.parametrize("kwargs,status", [
    ({"query": "   "}, 400),
    ({"query": "x" * 301}, 400),
    ({"track1": "other"}, 400),
])
def test_bad_searches_are_refused(video_bytes, kwargs, status):
    assert search(upload(video_bytes).json()["video_id"], **kwargs).status_code == status


def test_search_on_unknown_video_is_404():
    assert search("nope").status_code == 404


def test_bad_uploads_are_refused(video_bytes):
    assert upload(video_bytes, name="notes.txt").status_code == 400
    response = upload(b"this is not a video", name="fake.mp4")
    assert response.status_code == 400 and "could not be read" in response.json()["detail"]


def test_unknown_job_is_404():
    assert client.get("/api/jobs/nope").status_code == 404
