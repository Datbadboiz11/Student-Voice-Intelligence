from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference import find_project_dir, normalize_text
from src.reranker import FeedbackReranker, get_reranker


PAYLOAD_COLUMNS = [
    "source_dataset",
    "split_original",
    "sentiment_std_3class",
    "topic_group",
    "is_toxic",
    "urgency_level_final",
]


@dataclass(frozen=True)
class RetrievalConfig:
    project_dir: Path
    qdrant_url: str
    collection_name: str
    embedding_model_name: str
    data_path: Path
    rerank_top_n: int

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "RetrievalConfig":
        root = project_dir or find_project_dir()
        return cls(
            project_dir=root,
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "student_feedback"),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "keepitreal/vietnamese-sbert",
            ),
            data_path=Path(
                os.getenv(
                    "STUDENT_VOICE_RETRIEVAL_DATA",
                    root / "data/processed/student_voice_enriched_reviewed.csv",
                )
            ),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "20")),
        )


def _payload_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class SemanticSearchService:
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig.from_project()
        self._client: Any | None = None
        self._model: Any | None = None
        self._reranker: FeedbackReranker | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.config.qdrant_url, timeout=30)
        return self._client

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model_name)
        return self._model

    def _collection_exists(self) -> bool:
        return bool(self._get_client().collection_exists(self.config.collection_name))

    def _get_reranker(self) -> FeedbackReranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def health(self) -> dict[str, Any]:
        try:
            if not self._collection_exists():
                return {
                    "status": "not_indexed",
                    "qdrant_url": self.config.qdrant_url,
                    "collection": self.config.collection_name,
                }
            info = self._get_client().get_collection(self.config.collection_name)
            return {
                "status": "ready",
                "qdrant_url": self.config.qdrant_url,
                "collection": self.config.collection_name,
                "points_count": info.points_count,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "qdrant_url": self.config.qdrant_url,
                "collection": self.config.collection_name,
                "detail": str(exc),
            }

    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: str | None = None,
        sentiment: str | None = None,
        urgency: str | None = None,
        toxic: int | None = None,
    ) -> list[dict[str, Any]]:
        query = normalize_text(query)
        if not query:
            raise ValueError("Query must not be empty.")
        if not self._collection_exists():
            raise FileNotFoundError(
                "Semantic index does not exist. Run scripts/build_vector_index.py first."
            )

        from qdrant_client import models

        conditions = []
        for key, value in {
            "topic_group": topic,
            "sentiment_std_3class": sentiment,
            "urgency_level_final": urgency,
            "is_toxic": toxic,
        }.items():
            if value is not None:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        query_filter = models.Filter(must=conditions) if conditions else None

        vector = self._get_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        candidate_limit = max(top_k, self.config.rerank_top_n)
        response = self._get_client().query_points(
            collection_name=self.config.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )

        rows = []
        for point in response.points:
            payload = point.payload or {}
            rows.append(
                {
                    "id": str(point.id),
                    "vector_score": round(float(point.score), 6),
                    "text": payload.get("text", ""),
                    "source_dataset": payload.get("source_dataset"),
                    "sentiment": payload.get("sentiment_std_3class"),
                    "topic": payload.get("topic_group"),
                    "toxic": payload.get("is_toxic"),
                    "urgency": payload.get("urgency_level_final"),
                }
            )
        rerank_scores = self._get_reranker().score(query, [row["text"] for row in rows])
        for row, rerank_score in zip(rows, rerank_scores):
            row["rerank_score"] = round(rerank_score, 6)
        rows.sort(key=lambda row: row["rerank_score"], reverse=True)
        return rows[:top_k]

    def build_index(
        self,
        data_path: Path | None = None,
        recreate: bool = False,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        from qdrant_client import models

        source_path = data_path or self.config.data_path
        if not source_path.exists():
            raise FileNotFoundError(f"Retrieval data not found: {source_path}")

        frame = pd.read_csv(source_path)
        if "text" not in frame.columns:
            raise ValueError("Retrieval data must contain a 'text' column.")

        frame = frame.dropna(subset=["text"]).copy()
        frame["text"] = frame["text"].astype(str).map(normalize_text)
        frame = frame[frame["text"] != ""].drop_duplicates(subset=["text"])
        if frame.empty:
            raise ValueError("Retrieval data has no usable feedback text.")

        client = self._get_client()
        if self._collection_exists() and recreate:
            client.delete_collection(self.config.collection_name)

        model = self._get_model()
        if hasattr(model, "get_embedding_dimension"):
            vector_size = int(model.get_embedding_dimension())
        else:
            vector_size = int(model.get_sentence_embedding_dimension())
        if not self._collection_exists():
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        if client.count(collection_name=self.config.collection_name, exact=True).count:
            raise ValueError(
                "Collection already contains vectors. Use --recreate to rebuild the index."
            )

        total = len(frame)
        for start in range(0, total, batch_size):
            batch = frame.iloc[start : start + batch_size]
            texts = batch["text"].tolist()
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = []
            for row_number, (index, row) in enumerate(batch.iterrows()):
                record_id = str(row.get("row_id", index))
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{row.get('source_dataset', '')}|{record_id}|{row['text']}",
                    )
                )
                payload = {"record_id": record_id, "text": row["text"]}
                payload.update(
                    {
                        column: _payload_value(row[column])
                        for column in PAYLOAD_COLUMNS
                        if column in row.index
                    }
                )
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vectors[row_number].tolist(),
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )

        return {
            "collection": self.config.collection_name,
            "source_path": str(source_path),
            "indexed_rows": total,
            "embedding_model": self.config.embedding_model_name,
        }


@lru_cache(maxsize=1)
def get_retrieval_service() -> SemanticSearchService:
    return SemanticSearchService()
