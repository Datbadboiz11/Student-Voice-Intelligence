from __future__ import annotations

import os
from functools import lru_cache


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class FeedbackReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = os.getenv("RERANKER_DEVICE") or None
            self._model = CrossEncoder(self.model_name, max_length=256, device=device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._get_model().predict(
            pairs,
            batch_size=min(32, len(pairs)),
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


@lru_cache(maxsize=1)
def get_reranker() -> FeedbackReranker:
    return FeedbackReranker()
