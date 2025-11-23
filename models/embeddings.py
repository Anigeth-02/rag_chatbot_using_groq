# models/embeddings.py
"""
Embedding model wrapper using sentence-transformers (local, free).
Export:
- get_embeddings(texts: List[str]) -> List[List[float]]
- get_embedding_dimension() -> int
"""

from typing import List
import logging
from config.config import EMBEDDING_MODEL_NAME

_embedding_model = None
_embedding_dim = None


def _load_model():
    """
    Lazy-load the sentence-transformers model.
    """
    global _embedding_model, _embedding_dim
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Run a tiny test to get dimension
        test_vec = _embedding_model.encode(["test"])[0]
        _embedding_dim = len(test_vec)
        return _embedding_model
    except Exception as e:
        logging.exception("Failed to load embedding model: %s", e)
        _embedding_model = None
        _embedding_dim = 0
        return None


def get_embedding_dimension() -> int:
    """
    Return the embedding dimension for the current model.
    """
    global _embedding_dim
    if _embedding_dim is not None:
        return _embedding_dim

    _load_model()
    return _embedding_dim or 0


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Return embeddings for a list of strings as Python lists (for FAISS).
    """
    if not texts:
        return []

    model = _load_model()
    if model is None:
        logging.error("Embedding model is not available.")
        # return zero vectors as a fallback
        return [[0.0] * 384 for _ in texts]  # assume 384-dim by default

    try:
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
    except Exception as e:
        logging.exception("Failed to compute embeddings: %s", e)
        return [[0.0] * get_embedding_dimension() for _ in texts]
