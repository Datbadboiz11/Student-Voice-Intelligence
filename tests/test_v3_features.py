from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import api.app as api_app
from src.analytics import AnalyticsConfig, AnalyticsService
from src.reporting import ReportService
from src.reviews import ReviewConfig, ReviewService
from src.storage import AppStorage, StorageConfig
from src.topic_discovery import TopicDiscoveryConfig, TopicDiscoveryService


def make_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"row_id": "f1", "text": "wifi phòng học yếu và mất kết nối", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "facilities", "urgency_level_final": "high", "is_toxic": 0},
            {"row_id": "f2", "text": "máy chiếu phòng học quá mờ", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "facilities", "urgency_level_final": "medium", "is_toxic": 0},
            {"row_id": "f3", "text": "wifi phòng thực hành rất chậm", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "others", "urgency_level_final": "high", "is_toxic": 0},
            {"row_id": "f4", "text": "wifi phòng học thường xuyên bị rớt", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "others", "urgency_level_final": "medium", "is_toxic": 0},
            {"row_id": "f5", "text": "căng tin ít món ăn vào buổi trưa", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "others", "urgency_level_final": "low", "is_toxic": 0},
            {"row_id": "f6", "text": "căng tin phục vụ đồ ăn quá chậm", "source_dataset": "A", "sentiment_std_3class": "negative", "topic_group": "others", "urgency_level_final": "low", "is_toxic": 0},
            {"row_id": "f7", "text": "website đăng ký học phần thường lỗi", "source_dataset": "B", "sentiment_std_3class": "neutral", "topic_group": "others", "urgency_level_final": "medium", "is_toxic": 0},
            {"row_id": "f8", "text": "website học phần tải rất chậm", "source_dataset": "B", "sentiment_std_3class": "negative", "topic_group": "others", "urgency_level_final": "medium", "is_toxic": 0},
        ]
    ).to_csv(path, index=False)


def make_storage(tmp_path: Path) -> AppStorage:
    return AppStorage(StorageConfig(tmp_path / "state.db"))


def test_review_is_persisted_and_used_by_analytics(tmp_path):
    data_path = tmp_path / "feedback.csv"
    make_csv(data_path)
    storage = make_storage(tmp_path)
    review = ReviewService(ReviewConfig(data_path, StorageConfig(tmp_path / "state.db")), storage)

    pending = review.list_feedback(state="pending", urgency="high")
    assert pending["items"][0]["feedback_id"] == "f1"
    saved = review.save_review("f1", "low", reviewer="tester", note="Đã xử lý")

    assert saved["urgency_final"] == "low"
    analytics = AnalyticsService(AnalyticsConfig(tmp_path, data_path), storage).get_analytics(urgency="low")
    assert analytics["filtered_feedback"] == 3
    assert analytics["reviewed_feedback"] == 1


class FakeRetriever:
    def search(self, **kwargs):
        return [{"id": "f1", "text": "wifi phòng học yếu", "topic": kwargs["topic"], "sentiment": "negative", "urgency": "high", "rerank_score": 3.2}]


def test_report_is_grounded_and_saved(tmp_path):
    data_path = tmp_path / "feedback.csv"
    make_csv(data_path)
    storage = make_storage(tmp_path)
    analytics = AnalyticsService(AnalyticsConfig(tmp_path, data_path), storage)
    service = ReportService(analytics=analytics, retriever=FakeRetriever(), storage=storage)

    report = service.generate(title="Báo cáo thử", topic="facilities")

    assert report["title"] == "Báo cáo thử"
    assert "Feedback làm bằng chứng" in report["content_markdown"]
    assert report["data"]["evidence"][0]["id"] == "f1"
    assert service.get_report(report["id"])["title"] == "Báo cáo thử"


def test_topic_discovery_creates_reviewable_clusters(tmp_path):
    data_path = tmp_path / "feedback.csv"
    make_csv(data_path)
    storage = make_storage(tmp_path)
    service = TopicDiscoveryService(
        TopicDiscoveryConfig(data_path, StorageConfig(tmp_path / "state.db")), storage
    )

    result = service.run(topic="others", max_items=50, min_cluster_size=3)

    assert result["candidate_count"] == 6
    assert result["cluster_count"] >= 1
    cluster = result["clusters"][0]
    approved = service.approve_cluster(cluster["id"], "Hạ tầng số")
    assert approved["status"] == "approved"
    assert approved["approved_name"] == "Hạ tầng số"


class FakeReviewAPIService:
    def list_feedback(self, **_kwargs):
        return {"items": [], "count": 0, "state": "pending"}

    def save_review(self, feedback_id, urgency_final, reviewer, note):
        return {"feedback_id": feedback_id, "urgency_final": urgency_final, "reviewer": reviewer, "note": note}


class FakeReportAPIService:
    def generate(self, **kwargs):
        return {"id": 1, "title": kwargs["title"] or "Báo cáo", "content_markdown": "# Báo cáo"}

    def list_reports(self, _limit):
        return []

    def get_report(self, _report_id):
        return None


def test_review_and_report_api_routes(monkeypatch):
    monkeypatch.setattr(api_app, "get_review_service", lambda: FakeReviewAPIService())
    monkeypatch.setattr(api_app, "get_report_service", lambda: FakeReportAPIService())
    client = TestClient(api_app.app)

    review = client.post("/reviews/f1", json={"urgency_final": "high", "reviewer": "tester", "note": "Cần xử lý"})
    report = client.post("/reports/generate", json={"title": "Báo cáo API", "topic": "facilities"})

    assert review.status_code == 200
    assert review.json()["feedback_id"] == "f1"
    assert report.status_code == 200
    assert report.json()["title"] == "Báo cáo API"
