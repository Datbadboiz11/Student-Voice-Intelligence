from pathlib import Path

import pandas as pd

from src.analytics import AnalyticsConfig, AnalyticsService


def make_service(tmp_path: Path) -> AnalyticsService:
    path = tmp_path / "analytics.csv"
    pd.DataFrame(
        [
            {"source_dataset": "NEU_ESC", "sentiment_std_3class": "negative", "topic_group": "facilities", "urgency_level_final": "high", "is_toxic": 0},
            {"source_dataset": "NEU_ESC", "sentiment_std_3class": "negative", "topic_group": "facilities", "urgency_level_final": "low", "is_toxic": 1},
            {"source_dataset": "UIT_VSFC", "sentiment_std_3class": "positive", "topic_group": "teaching_learning", "urgency_level_final": "low", "is_toxic": 0},
            {"source_dataset": "UIT_VSFC", "sentiment_std_3class": "neutral", "topic_group": "student_services", "urgency_level_final": "medium", "is_toxic": 0},
        ]
    ).to_csv(path, index=False)
    return AnalyticsService(AnalyticsConfig(project_dir=tmp_path, data_path=path))


def test_analytics_returns_distributions_and_cross_tables(tmp_path):
    result = make_service(tmp_path).get_analytics()

    assert result["total_feedback"] == 4
    assert result["filtered_feedback"] == 4
    assert sum(row["count"] for row in result["sentiment_distribution"]) == 4
    assert result["negative_by_topic"] == [{"topic": "facilities", "count": 2}]
    assert sum(row["count"] for row in result["sentiment_topic_matrix"]) == 4


def test_analytics_applies_all_filters(tmp_path):
    result = make_service(tmp_path).get_analytics(
        dataset="NEU_ESC", topic="facilities", sentiment="negative", toxic=1
    )

    assert result["total_feedback"] == 4
    assert result["filtered_feedback"] == 1
    assert result["urgency_distribution"] == [{"label": "low", "count": 1, "percentage": 100.0}]
