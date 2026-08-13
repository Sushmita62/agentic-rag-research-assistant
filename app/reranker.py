"""Cross-encoder reranker. Bi-encoder + BM25 give the shortlist; this scores it well.

Cross-encoder = query and doc go through the model TOGETHER, so it can score
their joint relevance (unlike bi-encoders which embed each side alone). Best
quality lift in RAG for the cost.
"""
from __future__ import annotations

from app.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.reranker_model
        self._model = None                                     # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, str]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """candidates = [(id, text), ...]. Returns [(id, cross_encoder_score), ...] top_k."""
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip((c[0] for c in candidates), (float(s) for s in scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
