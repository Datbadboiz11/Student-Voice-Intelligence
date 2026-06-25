from src.chat import ChatService
from src.storage import AppStorage, StorageConfig


class FakeRag:
    def __init__(self):
        self.received = None

    def ask(self, **kwargs):
        self.received = kwargs
        return {
            "answer": "Wifi ph\u00f2ng h\u1ecdc y\u1ebfu. [1]",
            "grounded": True,
            "retrieved_count": 1,
            "evidence": [{"id": "f1", "text": "Wifi ph\u00f2ng h\u1ecdc y\u1ebfu."}],
        }


def test_chat_session_persists_messages_and_rag_evidence(tmp_path):
    storage = AppStorage(StorageConfig(tmp_path / "state.db"))
    rag = FakeRag()
    service = ChatService(storage=storage, rag=rag)

    session = service.create_session()
    response = service.ask(session["id"], "Sinh vi\u00ean ph\u00e0n n\u00e0n g\u00ec v\u1ec1 Wifi?", top_k=6)

    assert response is not None
    assert response["session"]["title"].startswith("Sinh vi\u00ean ph\u00e0n n\u00e0n")
    saved = service.get_session(session["id"])
    assert saved is not None
    assert [message["role"] for message in saved["messages"]] == ["user", "assistant"]
    assert saved["messages"][1]["result"]["evidence"][0]["id"] == "f1"
    assert rag.received["history"] == []


def test_chat_session_reuses_saved_history_and_can_be_deleted(tmp_path):
    storage = AppStorage(StorageConfig(tmp_path / "state.db"))
    rag = FakeRag()
    service = ChatService(storage=storage, rag=rag)
    session = service.create_session("Wifi")

    service.ask(session["id"], "Wifi y\u1ebfu?", top_k=6)
    service.ask(session["id"], "C\u00f2n v\u1ea5n \u0111\u1ec1 n\u00e0o n\u1eefa?", top_k=6)

    assert len(rag.received["history"]) == 2
    assert service.list_sessions()[0]["id"] == session["id"]
    assert service.delete_session(session["id"]) is True
    assert service.get_session(session["id"]) is None
