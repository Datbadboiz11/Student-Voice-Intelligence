from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from src.analytics import AnalyticsService, get_analytics_service
from src.retrieval import SemanticSearchService, get_retrieval_service
from src.storage import AppStorage


class ReportService:
    def __init__(
        self,
        analytics: AnalyticsService | Any | None = None,
        retriever: SemanticSearchService | Any | None = None,
        storage: AppStorage | None = None,
    ) -> None:
        self.analytics = analytics or get_analytics_service()
        self.retriever = retriever or get_retrieval_service()
        self.storage = storage or self.analytics.storage

    @staticmethod
    def _filter_values(filters: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in filters.items() if value not in (None, "")}

    @staticmethod
    def _count(rows: list[dict[str, Any]], label: str) -> int:
        return next((int(row["count"]) for row in rows if str(row["label"]) == label), 0)

    def _topic_evidence(self, topic: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return self.retriever.search(
                query=f"Sinh viên phản ánh gì về {topic}?",
                top_k=2,
                topic=topic,
                sentiment="negative",
                urgency=filters.get("urgency"),
                toxic=filters.get("toxic"),
            )
        except (ConnectionError, FileNotFoundError, ValueError):
            return []

    def generate(self, title: str | None = None, **filters: Any) -> dict[str, Any]:
        selected_filters = self._filter_values(filters)
        analytics = self.analytics.get_analytics(**selected_filters)
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        report_title = title.strip() if title and title.strip() else f"Báo cáo phản hồi sinh viên — {now}"
        total = analytics["filtered_feedback"]
        negative = self._count(analytics["sentiment_distribution"], "negative")
        high = self._count(analytics["urgency_distribution"], "high")
        evidence: list[dict[str, Any]] = []
        lines = [f"# {report_title}", "", f"_Tạo lúc {now}._", "", "## Tổng quan", ""]
        if not total:
            lines.extend(["Không có feedback phù hợp với bộ lọc đã chọn.", ""])
        else:
            lines.extend(
                [
                    f"- Phân tích **{total:,}** feedback.",
                    f"- Có **{negative:,}** feedback tiêu cực ({negative / total * 100:.1f}%).",
                    f"- Có **{high:,}** feedback ở mức khẩn cấp cao.",
                    "",
                    "## Vấn đề cần theo dõi",
                    "",
                ]
            )
            top_topics = analytics["negative_by_topic"][:5]
            if not top_topics:
                lines.append("Không có feedback tiêu cực trong phạm vi dữ liệu đã chọn.")
            for row in top_topics:
                topic = str(row["topic"])
                topic_count = int(row["count"])
                lines.append(f"### {topic}")
                lines.append(f"- Có **{topic_count:,}** feedback tiêu cực thuộc chủ đề này.")
                topic_evidence = self._topic_evidence(topic, selected_filters)
                for item in topic_evidence:
                    citation = len(evidence) + 1
                    evidence.append(
                        {
                            "citation": citation,
                            "id": item.get("id"),
                            "text": item.get("text", ""),
                            "topic": item.get("topic", topic),
                            "sentiment": item.get("sentiment"),
                            "urgency": item.get("urgency"),
                            "rerank_score": item.get("rerank_score"),
                        }
                    )
                    lines.append(f"- Feedback đại diện [{citation}].")
                lines.append("")
            lines.extend(["## Khuyến nghị", ""])
            if high:
                lines.append("- Ưu tiên kiểm tra các feedback có mức khẩn cấp cao trước.")
            if top_topics:
                lines.append("- Phân công người phụ trách rà soát các chủ đề có nhiều feedback tiêu cực nhất.")
            lines.append("- Dùng feedback đại diện bên dưới để xác minh nguyên nhân trước khi đưa ra quyết định.")
            lines.append("")
        lines.extend(["## Feedback làm bằng chứng", ""])
        if evidence:
            for item in evidence:
                lines.append(
                    f"[{item['citation']}] {item['topic']} | {item['sentiment']} | urgency {item['urgency']}"
                )
                lines.append(f"> {item['text']}")
                lines.append("")
        else:
            lines.append("Chưa có feedback đại diện phù hợp.")
            lines.append("")
        data = {"analytics": analytics, "evidence": evidence}
        return self.storage.save_report(report_title, selected_filters, "\n".join(lines), data)

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.storage.list_reports(limit)

    def get_report(self, report_id: int) -> dict[str, Any] | None:
        return self.storage.get_report(report_id)


@lru_cache(maxsize=1)
def get_report_service() -> ReportService:
    return ReportService()
