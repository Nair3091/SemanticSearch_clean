"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

FastAPI/Flask web app: video upload, query input, ranked-result / reject timeline display.
Calls into src.query_understanding, src.vision_retrieval, and src.fusion_confidence
via the orchestrator in scripts/run_pipeline.py — do not duplicate pipeline logic here.
"""

# from fastapi import FastAPI, UploadFile
#
# app = FastAPI()
#
# @app.post("/query")
# def run_query(video: UploadFile, query: str):
#     raise NotImplementedError
