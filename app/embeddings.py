"""bge-small-en-v1.5 wrapper. Batched, SQLite-cached by SHA256(text)."""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from sqlalchemy import select

from app.config import settings
from app.storage import EmbedCache, session


class Embedder:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model = None                                    # lazy: ~130MB download

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        hashes = [self._hash(t) for t in texts]
        cache: dict[str, np.ndarray] = {}

        with session() as s:
            rows = s.execute(select(EmbedCache).where(EmbedCache.hash.in_(hashes))).scalars().all()
            for r in rows:
                cache[r.hash] = np.frombuffer(r.vec, dtype=np.float32)

        missing = [(i, h, texts[i]) for i, h in enumerate(hashes) if h not in cache]

        if missing:
            model = self._load()
            new_vecs = model.encode(
                [t for _, _, t in missing],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            with session() as s:
                for (_, h, _), v in zip(missing, new_vecs):
                    cache[h] = v
                    s.merge(EmbedCache(hash=h, vec=v.tobytes()))

        return np.stack([cache[h] for h in hashes])
