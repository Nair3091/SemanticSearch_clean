import os
import numpy as np
import faiss


def build_index(frame_embeddings: np.ndarray, index_path: str) -> None:
    """
    Build a FAISS index from frame embeddings and save it to disk.

    frame_embeddings: (N, D) array, L2-normalized (as returned by
                       clip_encoder.embed_frames).
    """
    if frame_embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D (N, D) array, got shape {frame_embeddings.shape}")

    dim = frame_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(frame_embeddings.astype(np.float32))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def load_index(index_path: str) -> faiss.Index:
    """
    Load a precomputed FAISS index from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found at {index_path}")
    return faiss.read_index(index_path)


def search(index: faiss.Index, query_embedding: np.ndarray, top_k: int):
    """
    Return the top_k nearest frame embeddings for a given query embedding.

    query_embedding: (D,) or (1, D) array, L2-normalized.
    Returns (scores, indices): both (top_k,) arrays — scores are cosine
    similarities (since the index is IndexFlatIP on normalized vectors),
    indices are positions into the original frame_embeddings array
    passed to build_index.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    return scores[0], indices[0]
