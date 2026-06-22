from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from src.retrieval import RetrievalConfig, SemanticSearchService


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            text = text.lower()
            if "wifi" in text or "phong hoc" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.array(vectors, dtype=np.float32)


class FakeReranker:
    def score(self, _query, texts):
        return [1.0 if "Wifi" in text else 0.0 for text in texts]


def test_build_and_search_qdrant_index(tmp_path: Path):
    source_path = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "row_id": 1,
                "text": "Wifi phong hoc qua yeu.",
                "source_dataset": "UIT_VSFC",
                "sentiment_std_3class": "negative",
                "topic_group": "facilities",
                "is_toxic": 0,
                "urgency_level_final": "medium",
            },
            {
                "row_id": 2,
                "text": "Giang vien day de hieu.",
                "source_dataset": "UIT_VSFC",
                "sentiment_std_3class": "positive",
                "topic_group": "teaching_learning",
                "is_toxic": 0,
                "urgency_level_final": "low",
            },
        ]
    ).to_csv(source_path, index=False)

    config = RetrievalConfig(
        project_dir=tmp_path,
        qdrant_url="http://unused",
        collection_name="test_feedback",
        embedding_model_name="fake-model",
        data_path=source_path,
        rerank_top_n=20,
    )
    service = SemanticSearchService(config)
    service._client = QdrantClient(location=":memory:")
    service._model = FakeEmbeddingModel()
    service._reranker = FakeReranker()

    summary = service.build_index(recreate=True)
    results = service.search("wifi rat yeu", top_k=1, topic="facilities")

    assert summary["indexed_rows"] == 2
    assert len(results) == 1
    assert results[0]["vector_score"] == 1.0
    assert results[0]["rerank_score"] == 1.0
    assert results[0]["text"] == "Wifi phong hoc qua yeu."
    assert results[0]["source_dataset"] == "UIT_VSFC"
    assert results[0]["sentiment"] == "negative"
    assert results[0]["topic"] == "facilities"
    assert results[0]["toxic"] == 0
    assert results[0]["urgency"] == "medium"
