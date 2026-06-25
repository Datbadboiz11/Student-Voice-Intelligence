from __future__ import annotations

import hashlib
import uuid
from functools import lru_cache
from typing import Any

from src.inference import get_inference_service, normalize_text
from src.retrieval import get_retrieval_service
from src.storage import AppStorage

VALID_STATUS = {"new", "in_progress", "resolved"}


class FeedbackService:
    def __init__(self, storage: AppStorage | None = None, inference: Any | None = None, retriever: Any | None = None) -> None:
        self.storage = storage or AppStorage()
        self.inference = inference or get_inference_service()
        self.retriever = retriever or get_retrieval_service()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()

    def create(self, text: str, location: str = "", source_dataset: str = "admin_manual") -> tuple[dict[str, Any], bool]:
        text = normalize_text(text)
        if not text:
            raise ValueError("Feedback text must not be empty.")
        content_hash = self._hash(text)
        existing = self.storage.get_admin_feedback_by_hash(source_dataset, content_hash)
        if existing:
            return existing, False
        prediction = self.inference.analyze(text)
        record = {
            "id": str(uuid.uuid4()), "text": text, "location": normalize_text(location), "source_dataset": source_dataset,
            "content_hash": content_hash, "sentiment": prediction["sentiment"], "sentiment_confidence": prediction["sentiment_confidence"],
            "topic": prediction["topic"], "topic_confidence": prediction["topic_confidence"], "toxic": int(prediction["toxic"]),
            "urgency": prediction["urgency"], "status": "new",
        }
        saved = self.storage.save_admin_feedback(record)
        try:
            self.retriever.upsert_admin_feedback(saved)
        except Exception:
            self.storage.delete_admin_feedback(saved["id"])
            raise
        return saved, True

    def import_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        created, duplicates, errors = [], 0, []
        for index, row in enumerate(rows, start=1):
            try:
                item, is_new = self.create(str(row.get("text", "")), str(row.get("location", "")), "admin_csv")
                if is_new: created.append(item)
                else: duplicates += 1
            except Exception as exc:
                errors.append({"row": index, "detail": str(exc)})
        return {"created": len(created), "duplicates": duplicates, "errors": errors, "items": created}

    def list(self, status: str | None = None, topic: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.storage.list_admin_feedbacks(status, topic, limit)

    def update_status(self, feedback_id: str, status: str) -> dict[str, Any] | None:
        if status not in VALID_STATUS:
            raise ValueError("status must be new, in_progress, or resolved.")
        return self.storage.update_admin_feedback_status(feedback_id, status)

    def delete(self, feedback_id: str) -> bool:
        existing = next((row for row in self.list(limit=100_000) if row["id"] == feedback_id), None)
        if not existing: return False
        self.retriever.delete_admin_feedback(feedback_id)
        return self.storage.delete_admin_feedback(feedback_id)


@lru_cache(maxsize=1)
def get_feedback_service() -> FeedbackService:
    return FeedbackService()
