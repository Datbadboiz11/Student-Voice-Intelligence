from src.feedbacks import FeedbackService
from src.storage import AppStorage, StorageConfig


class FakeInference:
    def analyze(self, text):
        return {"sentiment": "negative", "sentiment_confidence": 0.9, "topic": "facilities", "topic_confidence": 0.8, "toxic": 0, "urgency": "medium"}


class FakeRetriever:
    def __init__(self): self.upserted = []; self.deleted = []
    def upsert_admin_feedback(self, record): self.upserted.append(record["id"])
    def delete_admin_feedback(self, feedback_id): self.deleted.append(feedback_id)


def test_imported_feedback_is_persisted_deduplicated_and_indexed(tmp_path):
    retriever = FakeRetriever()
    service = FeedbackService(AppStorage(StorageConfig(tmp_path / "state.db")), FakeInference(), retriever)
    item, created = service.create("Wifi phòng học rất yếu.", "A101")
    duplicate, created_again = service.create("Wifi phòng học rất yếu.", "A101")

    assert created is True
    assert created_again is False
    assert duplicate["id"] == item["id"]
    assert retriever.upserted == [item["id"]]
    assert service.update_status(item["id"], "resolved")["status"] == "resolved"
    assert service.delete(item["id"]) is True
    assert retriever.deleted == [item["id"]]
