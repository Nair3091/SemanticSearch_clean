import threading
from typing import List
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.common.config import load_config
from src.common.types import Frame

_cfg = load_config()
_MODEL_NAME = _cfg["vision_retrieval"]["clip_model_name"]
_requested = _cfg["hardware"]["device"]
if _requested in ("auto", "cuda"):
    # "auto" (and "cuda") use an NVIDIA GPU when PyTorch can see one, otherwise the CPU
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if _requested == "cuda" and _DEVICE == "cpu":
        print("WARNING: cuda requested in config but not available, falling back to CPU")
else:
    _DEVICE = _requested


def device() -> str:
    """Where CLIP runs: "cuda" or "cpu"."""
    return _DEVICE

_model = None
_processor = None
_load_lock = threading.Lock()   # video preparation and searches may load the model at the same time


def _load_model():
    global _model, _processor
    with _load_lock:
        if _model is None:
            model = CLIPModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
            model.eval()
            gpu = f" ({torch.cuda.get_device_name(0)})" if _DEVICE == "cuda" else ""
            print(f"[Track2/CLIP] {_MODEL_NAME} on {_DEVICE}{gpu}")
            _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
            _model = model
    return _model, _processor


def embedding_dim() -> int:
    model, _ = _load_model()
    return model.config.projection_dim


def embed_frames(frames: List[Frame], batch_size: int = 32) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((0, embedding_dim()), dtype=np.float32)

    model, processor = _load_model()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            images = [f.image for f in batch]
            inputs = processor(images=images, return_tensors="pt").to(_DEVICE)
            output = model.get_image_features(**inputs)
            # transformers 5.x wraps the embedding in .pooler_output;
            # transformers 4.x returns the tensor directly. Handle both.
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def embed_text(sub_queries: List[str]) -> np.ndarray:
    """
    Run the frozen CLIP text encoder over sub-query strings.
    Returns an (M, D) array of L2-normalized text embeddings.
    """
    if len(sub_queries) == 0:
        return np.zeros((0, embedding_dim()), dtype=np.float32)

    model, processor = _load_model()

    with torch.no_grad():
        inputs = processor(text=sub_queries, return_tensors="pt", padding=True).to(_DEVICE)
        output = model.get_text_features(**inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features.cpu().numpy()


def score_frames(frame_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    if frame_embeddings.shape[0] == 0 or text_embeddings.shape[0] == 0:
        return np.zeros((frame_embeddings.shape[0], text_embeddings.shape[0]), dtype=np.float32)

    return frame_embeddings @ text_embeddings.T