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


def post(video_bytes, query="a dog", name="v.avi", track1="rules"):
    return client.post("/api/search", data={"query": query, "track1": track1},
                       files={"video": (name, video_bytes, "video/avi")})


def test_page_and_status_load():
    assert "SemanticSearch" in client.get("/").text
    assert {"negation_ready", "negation_reason"} <= set(client.get("/api/status").json())


def test_search_returns_timestamp_and_deletes_the_upload(video_bytes, monkeypatch):
    seen = {}

    def fake_run(path, query, use_qwen, check_negation, progress):
        seen.update(path=path, query=query, use_qwen=use_qwen)
        progress("track1", {"state": "done", "seconds": 0.1})
        return fake_report(query)

    monkeypatch.setattr(web.run_pipeline, "run", fake_run)
    job = wait_for(post(video_bytes).json()["job_id"])
    assert job["state"] == "done", job["error"]
    assert job["result"]["accepted"] and job["result"]["best"]["start"] == 1.0
    assert job["result"]["duration"] == pytest.approx(3.0)
    assert job["tracks"]["track1"]["state"] == "done"
    assert seen["query"] == "a dog" and seen["use_qwen"] is False
    assert not Path(seen["path"]).exists()


def test_pipeline_error_reaches_the_page(video_bytes, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("ConCLIP weights not found")

    monkeypatch.setattr(web.run_pipeline, "run", boom)
    job = wait_for(post(video_bytes).json()["job_id"])
    assert job["state"] == "error" and "ConCLIP weights" in job["error"]


@pytest.mark.parametrize("kwargs,status", [
    ({"query": "   "}, 400),
    ({"query": "x" * 301}, 400),
    ({"track1": "other"}, 400),
    ({"name": "notes.txt"}, 400),
])
def test_bad_requests_are_refused(video_bytes, kwargs, status):
    assert post(video_bytes, **kwargs).status_code == status


def test_file_that_is_not_a_video_is_refused():
    response = post(b"this is not a video", name="fake.mp4")
    assert response.status_code == 400 and "could not be read" in response.json()["detail"]


def test_unknown_job_is_404():
    assert client.get("/api/jobs/nope").status_code == 404
