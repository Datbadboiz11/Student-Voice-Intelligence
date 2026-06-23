from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference import find_project_dir


ANALYTICS_COLUMNS = {
    "source_dataset": "dataset",
    "sentiment_std_3class": "sentiment",
    "topic_group": "topic",
    "urgency_level_final": "urgency",
    "is_toxic": "toxic",
}


@dataclass(frozen=True)
class AnalyticsConfig:
    project_dir: Path
    data_path: Path

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "AnalyticsConfig":
        root = project_dir or find_project_dir()
        return cls(
            project_dir=root,
            data_path=Path(
                os.getenv(
                    "STUDENT_VOICE_ANALYTICS_DATA",
                    root / "data/processed/student_voice_enriched_reviewed.csv",
                )
            ),
        )


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {key: int(value) if isinstance(value, int) else value for key, value in row.items()}
        for row in frame[columns].to_dict(orient="records")
    ]


class AnalyticsService:
    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self.config = config or AnalyticsConfig.from_project()
        self._frame: pd.DataFrame | None = None

    def _get_frame(self) -> pd.DataFrame:
        if self._frame is None:
            if not self.config.data_path.exists():
                raise FileNotFoundError(f"Analytics data not found: {self.config.data_path}")

            frame = pd.read_csv(self.config.data_path, usecols=list(ANALYTICS_COLUMNS))
            frame = frame.rename(columns=ANALYTICS_COLUMNS)
            for column in ("dataset", "sentiment", "topic", "urgency"):
                frame[column] = frame[column].fillna("unknown").astype(str)
            frame["toxic"] = pd.to_numeric(frame["toxic"], errors="coerce").fillna(0).astype(int)
            self._frame = frame
        return self._frame

    @staticmethod
    def _distribution(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        total = len(frame)
        counts = frame[column].value_counts().rename_axis("label").reset_index(name="count")
        counts["percentage"] = (counts["count"] / total * 100).round(2) if total else 0.0
        return _records(counts, ["label", "count", "percentage"])

    @staticmethod
    def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        grouped = frame.groupby(columns, dropna=False).size().reset_index(name="count")
        return _records(grouped.sort_values("count", ascending=False), [*columns, "count"])

    def get_analytics(
        self,
        dataset: str | None = None,
        topic: str | None = None,
        sentiment: str | None = None,
        urgency: str | None = None,
        toxic: int | None = None,
    ) -> dict[str, Any]:
        source = self._get_frame()
        frame = source
        for column, value in {
            "dataset": dataset,
            "topic": topic,
            "sentiment": sentiment,
            "urgency": urgency,
            "toxic": toxic,
        }.items():
            if value is not None:
                frame = frame[frame[column] == value]

        negative = frame[frame["sentiment"] == "negative"]
        return {
            "total_feedback": int(len(source)),
            "filtered_feedback": int(len(frame)),
            "source_distribution": self._distribution(frame, "dataset"),
            "sentiment_distribution": self._distribution(frame, "sentiment"),
            "topic_distribution": self._distribution(frame, "topic"),
            "urgency_distribution": self._distribution(frame, "urgency"),
            "toxic_distribution": self._distribution(frame, "toxic"),
            "negative_by_topic": self._group_counts(negative, ["topic"]),
            "urgency_by_topic": self._group_counts(frame, ["topic", "urgency"]),
            "sentiment_topic_matrix": self._group_counts(frame, ["topic", "sentiment"]),
        }


@lru_cache(maxsize=1)
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()
