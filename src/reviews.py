from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics import AnalyticsConfig
from src.storage import AppStorage, StorageConfig


VALID_URGENCY = {"low", "medium", "high"}


@dataclass(frozen=True)
class ReviewConfig:
    data_path: Path
    storage_config: StorageConfig

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "ReviewConfig":
        analytics = AnalyticsConfig.from_project(project_dir)
        return cls(data_path=analytics.data_path, storage_config=StorageConfig.from_project(project_dir))


class ReviewService:
    def __init__(
        self,
        config: ReviewConfig | None = None,
        storage: AppStorage | None = None,
    ) -> None:
        self.config = config or ReviewConfig.from_project()
        self.storage = storage or AppStorage(self.config.storage_config)

    def _feedback_frame(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Review data not found: {self.config.data_path}")
        headers = pd.read_csv(self.config.data_path, nrows=0).columns
        available = [
            column
            for column in ("row_id", "text", "source_dataset", "topic_group", "sentiment_std_3class", "urgency_level_final")
            if column in headers
        ]
        frame = pd.read_csv(self.config.data_path, usecols=available)
        if "row_id" not in frame:
            frame["row_id"] = frame.index.astype(str)
        frame["row_id"] = frame["row_id"].astype(str)
        for source, target in {
            "source_dataset": "dataset",
            "topic_group": "topic",
            "sentiment_std_3class": "sentiment",
            "urgency_level_final": "urgency_predicted",
        }.items():
            frame[target] = frame[source] if source in frame else "unknown"
        frame["text"] = frame["text"].fillna("").astype(str) if "text" in frame else ""
        return frame[["row_id", "text", "dataset", "topic", "sentiment", "urgency_predicted"]]

    def list_feedback(
        self,
        state: str = "pending",
        urgency: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        if state not in {"pending", "reviewed", "all"}:
            raise ValueError("state must be pending, reviewed, or all.")
        frame = self._feedback_frame()
        reviews = self.storage.review_map()
        frame["review"] = frame["row_id"].map(reviews)
        frame["reviewed"] = frame["review"].notna()
        frame["urgency"] = frame.apply(
            lambda row: row["review"]["urgency_final"] if isinstance(row["review"], dict) else row["urgency_predicted"],
            axis=1,
        )
        if state == "pending":
            frame = frame[~frame["reviewed"]]
        elif state == "reviewed":
            frame = frame[frame["reviewed"]]
        if urgency:
            frame = frame[frame["urgency"] == urgency]
        priority = {"high": 0, "medium": 1, "low": 2}
        frame["priority"] = frame["urgency"].map(priority).fillna(3)
        frame = frame.sort_values(["priority", "row_id"]).head(limit)
        records = []
        for row in frame.to_dict(orient="records"):
            review = row.get("review") if isinstance(row.get("review"), dict) else {}
            records.append(
                {
                    "feedback_id": row["row_id"],
                    "text": row["text"],
                    "dataset": row["dataset"],
                    "topic": row["topic"],
                    "sentiment": row["sentiment"],
                    "urgency_predicted": row["urgency_predicted"],
                    "urgency_final": row["urgency"],
                    "reviewed": bool(row["reviewed"]),
                    "reviewer": review.get("reviewer"),
                    "note": review.get("note", ""),
                    "reviewed_at": review.get("reviewed_at"),
                }
            )
        return {"items": records, "count": len(records), "state": state}

    def save_review(
        self,
        feedback_id: str,
        urgency_final: str,
        reviewer: str = "admin",
        note: str = "",
    ) -> dict[str, Any]:
        if urgency_final not in VALID_URGENCY:
            raise ValueError("urgency_final must be low, medium, or high.")
        feedback_id = feedback_id.strip()
        if not feedback_id:
            raise ValueError("feedback_id must not be empty.")
        if len(reviewer.strip()) > 80 or len(note.strip()) > 500:
            raise ValueError("reviewer or note is too long.")
        return self.storage.save_review(feedback_id, urgency_final, reviewer.strip() or "admin", note.strip())


@lru_cache(maxsize=1)
def get_review_service() -> ReviewService:
    return ReviewService()
