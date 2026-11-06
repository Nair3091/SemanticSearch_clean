"""
Track 3 — Fusion, Confidence & Web Demo
Owner: Ihsal Riyas

Negation filter (ConCLIP).

Input  : the original video + Track 2's candidate segments (start/end timestamps)
         + the concepts that must be absent, from Track 1 (e.g. "a dog").
Output : for every segment, PASS / FAIL + a negation score.

How a segment is checked:
  1. Sample frames inside [start_time, end_time] every `sample_interval` seconds.
  2. Encode each excluded concept with the ConCLIP text encoder (once per query).
  3. Encode the frames with the ConCLIP image encoder (batched).
  4. cosine_similarity(frame, concept) < frame_threshold  -> the concept is absent from that frame.
  5. A frame only passes if EVERY excluded concept is absent from it.
  6. The segment PASSES if the fraction of passing frames >= min_pass_ratio.

Why the concept on its own, not a caption such as "a cat without a dog"? On a test video of known
scenes, similarity to the caption barely changed whether the excluded thing was visible or not
(best possible cutoff: 11 of 19 cases right), while similarity to the concept alone separated
them (19 of 19). The threshold comes from those same cases, so it still needs a proper calibration.

Why a ratio and not "every frame must pass"? It absorbs single-frame noise while still
rejecting segments where the excluded object is clearly visible.

This module does not import Track 1 or Track 2 code (README ground rule 3);
it does its own lightweight frame decoding with OpenCV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from src.common.types import CandidateSegment


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SegmentNegationResult:
    segment: CandidateSegment
    verdict: str                     # "PASS" | "FAIL" | "SKIP" | "ERROR"
    negation_score: Optional[float]  # 1 - mean similarity to the excluded concept: higher = more clearly absent
    pass_ratio: float
    frames_checked: int
    frame_timestamps: List[float] = field(default_factory=list)
    frame_scores: List[float] = field(default_factory=list)   # per frame, similarity to the closest concept
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict in ("PASS", "SKIP")

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.segment.start_time, 3),
            "end_time": round(self.segment.end_time, 3),
            "retrieval_confidence": round(float(self.segment.fused_confidence), 4),
            "verdict": self.verdict,
            "negation_score": None if self.negation_score is None else round(self.negation_score, 4),
            "pass_ratio": round(self.pass_ratio, 3),
            "frames_checked": self.frames_checked,
            "frame_timestamps": [round(t, 2) for t in self.frame_timestamps],
            "frame_scores": [round(s, 4) for s in self.frame_scores],
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Pure decision logic (no model needed -> unit-testable)
# ---------------------------------------------------------------------------

def decide_segment(sim_matrix: np.ndarray, frame_threshold: float, min_pass_ratio: float):
    """
    sim_matrix: (F, C) ConCLIP cosine similarities, F frames x C excluded concepts.
    Returns (verdict, negation_score, pass_ratio, per_frame_scores).
    """
    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
    if sim_matrix.ndim == 1:
        sim_matrix = sim_matrix.reshape(-1, 1)
    if sim_matrix.size == 0:
        return "ERROR", None, 0.0, []

    per_frame = sim_matrix.max(axis=1)                # the closest concept decides: ALL must be absent
    frame_pass = per_frame < frame_threshold
    pass_ratio = float(frame_pass.mean())
    verdict = "PASS" if pass_ratio >= min_pass_ratio else "FAIL"
    return verdict, float(1.0 - per_frame.mean()), pass_ratio, per_frame.tolist()


def segment_sample_times(start: float, end: float, interval: float, max_frames: int) -> List[float]:
    """Evenly spaced timestamps inside [start, end]; always at least one (the midpoint)."""
    if end <= start:
        return [max(0.0, start)]
    n = int((end - start) / interval) + 1 if interval > 0 else 1
    n = max(1, min(n, max_frames))
    if n == 1:
        return [(start + end) / 2.0]
    return [float(t) for t in np.linspace(start, end, n)]


# ---------------------------------------------------------------------------
# ConCLIP model wrapper
# ---------------------------------------------------------------------------

class ConCLIPNegationFilter:
    def __init__(self,
                 weights_path: str,
                 model_name: str = "ViT-B-32",
                 frame_threshold: float = 0.205,
                 min_pass_ratio: float = 0.7,
                 sample_interval: float = 1.0,
                 max_frames_per_segment: int = 16,
                 batch_size: int = 16,
                 device: Optional[str] = None):
        import torch
        import open_clip

        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"ConCLIP weights not found at '{weights_path}'. Set "
                f"fusion_confidence.negation.conclip_weights_path in configs/config.yaml "
                f"(file: conclip_b32_openclip_version.pt from the NegBench Google Drive).")

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.frame_threshold = frame_threshold
        self.min_pass_ratio = min_pass_ratio
        self.sample_interval = sample_interval
        self.max_frames = max_frames_per_segment
        self.batch_size = batch_size

        print(f"[Track3/ConCLIP] loading {model_name} weights from {weights_path} on {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weights_path, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    # -- encoders ---------------------------------------------------------

    def encode_text(self, captions: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tokens = self.tokenizer(list(captions)).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images_rgb), self.batch_size):
                batch = [self.preprocess(Image.fromarray(img)) for img in images_rgb[i:i + self.batch_size]]
                x = torch.stack(batch).to(self.device)
                feats = self.model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    # -- frame decoding -----------------------------------------------------

    @staticmethod
    def read_frames(video_path: str, timestamps: Sequence[float]):
        """Returns (timestamps_actually_read, list_of_RGB_frames)."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        got_t, got_f = [], []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            got_t.append(float(t))
            got_f.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return got_t, got_f

    # -- main entry -----------------------------------------------------------

    def verify_segments(self, video_path: str, segments: Sequence[CandidateSegment],
                        excluded_concepts: Sequence[str]) -> List[SegmentNegationResult]:
        if not excluded_concepts:
            return [SegmentNegationResult(seg, "SKIP", None, 1.0, 0, message="no negation in query")
                    for seg in segments]

        text_emb = self.encode_text(excluded_concepts)            # (C, D)
        results = []
        for seg in segments:
            times = segment_sample_times(seg.start_time, seg.end_time, self.sample_interval, self.max_frames)
            got_t, frames = self.read_frames(video_path, times)
            if not frames:
                results.append(SegmentNegationResult(seg, "ERROR", None, 0.0, 0,
                                                     message="could not decode any frame in segment"))
                continue
            img_emb = self.encode_images(frames)                     # (F, D)
            sims = img_emb @ text_emb.T                              # (F, C) cosine similarity
            verdict, score, ratio, per_frame = decide_segment(sims, self.frame_threshold, self.min_pass_ratio)
            results.append(SegmentNegationResult(
                segment=seg, verdict=verdict, negation_score=score, pass_ratio=ratio,
                frames_checked=len(frames), frame_timestamps=got_t, frame_scores=per_frame,
                message=f"{int(round(ratio * len(frames)))}/{len(frames)} frames satisfy the negation"))
        return results
