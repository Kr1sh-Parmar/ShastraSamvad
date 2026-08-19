"""One embedding model, loaded once, shared by retriever / dialogue manager / verifier."""
import numpy as np

from . import config

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def encode(texts: str | list[str]) -> np.ndarray:
    """Always returns 2-D float32, L2-normalized (so dot product == cosine)."""
    if isinstance(texts, str):
        texts = [texts]
    return model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
