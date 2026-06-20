"""
embeddings.py — thin shared wrapper around BAAI/bge-small-en-v1.5 via
sentence-transformers, used by both verifier.py (context-recall scoring)
and dedup.py (near-duplicate detection).

Not listed as its own file in the spec's File Structure, but both consumers
need the identical model/loading logic, and the spec is explicit that
bge-small is used for two distinct purposes (3a and 3b) — centralizing the
model load avoids loading the same local model twice in one process and
keeps the embedding logic in one place if the model name ever changes in
config.yaml.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG, setup_logging

logger = setup_logging(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model — avoids paying the load
    cost if a module imports this file but never actually embeds anything."""
    global _model
    if _model is None:
        emb_cfg = CONFIG["embedding"]
        logger.info(f"Loading embedding model {emb_cfg['model_name']} on {emb_cfg['device']}")
        _model = SentenceTransformer(emb_cfg["model_name"], device=emb_cfg["device"])
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, returning an (N, dim) normalized array.
    Normalizing here means downstream cosine similarity is just a dot
    product, which both verifier.py and dedup.py rely on."""
    if not texts:
        return np.zeros((0, 384))  # bge-small-en-v1.5 dim
    emb_cfg = CONFIG["embedding"]
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=emb_cfg["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two already-normalized vectors (dot product).
    Falls back to explicit normalization if vectors aren't unit length,
    in case this is ever called with non-normalized input."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
