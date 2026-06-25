from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics import AnalyticsConfig
from src.storage import AppStorage, StorageConfig


@dataclass(frozen=True)
class TopicDiscoveryConfig:
    data_path: Path
    storage_config: StorageConfig

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "TopicDiscoveryConfig":
        analytics = AnalyticsConfig.from_project(project_dir)
        return cls(analytics.data_path, StorageConfig.from_project(project_dir))


class TopicDiscoveryService:
    def __init__(
        self,
        config: TopicDiscoveryConfig | None = None,
        storage: AppStorage | None = None,
    ) -> None:
        self.config = config or TopicDiscoveryConfig.from_project()
        self.storage = storage or AppStorage(self.config.storage_config)

    def _load_candidates(self, topic: str | None, max_items: int) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Topic discovery data not found: {self.config.data_path}")
        headers = pd.read_csv(self.config.data_path, nrows=0).columns
        available = [column for column in ("row_id", "text", "topic_group") if column in headers]
        if "text" not in available:
            raise ValueError("Topic discovery data must contain a text column.")
        frame = pd.read_csv(self.config.data_path, usecols=available)
        if "row_id" not in frame:
            frame["row_id"] = frame.index.astype(str)
        imported = self.storage.list_admin_feedbacks(topic=topic, limit=max_items)
        if imported:
            imported_frame = pd.DataFrame(imported).rename(columns={"id": "row_id", "topic": "topic_group"})[["row_id", "text", "topic_group"]]
            frame = pd.concat([frame, imported_frame], ignore_index=True)
        if topic and "topic_group" in frame:
            frame = frame[frame["topic_group"].fillna("unknown").astype(str) == topic]
        frame = frame.dropna(subset=["text"]).copy()
        frame["text"] = frame["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        frame = frame[frame["text"].str.len() >= 8].drop_duplicates(subset=["text"])
        return frame.head(max_items).reset_index(drop=True)

    @staticmethod
    def _cluster_count(size: int, min_cluster_size: int) -> int:
        return max(2, min(12, size // max(min_cluster_size * 2, 1)))

    def run(
        self,
        topic: str | None = "others",
        max_items: int = 1200,
        min_cluster_size: int = 6,
    ) -> dict[str, Any]:
        if not 50 <= max_items <= 3000:
            raise ValueError("max_items must be between 50 and 3000.")
        if not 3 <= min_cluster_size <= 50:
            raise ValueError("min_cluster_size must be between 3 and 50.")
        frame = self._load_candidates(topic, max_items)
        if len(frame) < min_cluster_size * 2:
            raise ValueError("Not enough feedback for topic discovery.")
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=5000,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(frame["text"])
        if matrix.shape[1] == 0:
            raise ValueError("Feedback text does not contain enough distinct terms.")
        cluster_count = self._cluster_count(len(frame), min_cluster_size)
        model = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init=10, batch_size=256)
        labels = model.fit_predict(matrix)
        frame["cluster"] = labels
        terms = vectorizer.get_feature_names_out()
        clusters: list[dict[str, Any]] = []
        for label in sorted(set(labels)):
            members = frame[frame["cluster"] == label].copy()
            if len(members) < min_cluster_size:
                continue
            center = model.cluster_centers_[label]
            keywords = [terms[index] for index in center.argsort()[-4:][::-1] if len(terms[index]) > 2]
            scores = matrix[members.index].dot(center)
            members["score"] = scores
            examples = members.sort_values("score", ascending=False)["text"].head(3).tolist()
            clusters.append(
                {
                    "suggested_name": " / ".join(keywords[:3]) or f"Cụm phản ánh {label + 1}",
                    "keywords": keywords,
                    "examples": examples,
                    "feedback_ids": members["row_id"].astype(str).head(100).tolist(),
                    "size": int(len(members)),
                }
            )
        clusters.sort(key=lambda item: item["size"], reverse=True)
        result = self.storage.save_discovery_run(topic, len(frame), clusters)
        result["clusters"] = [
            cluster
            for cluster in self.storage.list_clusters(limit=len(clusters) + 20)
            if cluster["run_id"] == result["run_id"]
        ]
        return result

    def list_clusters(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.storage.list_clusters(status, limit)

    def approve_cluster(self, cluster_id: int, name: str) -> dict[str, Any] | None:
        name = name.strip()
        if not name:
            raise ValueError("Topic name must not be empty.")
        if len(name) > 120:
            raise ValueError("Topic name is too long.")
        return self.storage.update_cluster(cluster_id, "approved", name)

    def reject_cluster(self, cluster_id: int) -> dict[str, Any] | None:
        return self.storage.update_cluster(cluster_id, "rejected")


@lru_cache(maxsize=1)
def get_topic_discovery_service() -> TopicDiscoveryService:
    return TopicDiscoveryService()
