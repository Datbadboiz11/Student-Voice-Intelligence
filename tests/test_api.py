from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.app as api_app


class FakeInferenceService:
    def health(self):
        return {
            "status": "ready",
            "device": "cpu",
            "has_underthesea": True,
            "has_toxic_baseline": True,
            "has_urgency_baseline": True,
            "sentiment_labels": {"0": "negative", "1": "neutral", "2": "positive"},
            "topic_labels": {"0": "facility", "1": "lecturer"},
        }

    def analyze(self, text: str):
        if not text.strip():
            raise ValueError("Text must not be empty.")
        return {
            "text": text.strip(),
            "sentiment": "negative",
            "sentiment_confidence": 0.91,
            "sentiment_probabilities": {"negative": 0.91, "neutral": 0.06, "positive": 0.03},
            "topic": "facility",
            "topic_confidence": 0.88,
            "topic_probabilities": {"facility": 0.88, "lecturer": 0.12},
            "toxic": 0,
            "urgency": "medium",
        }

    def analyze_many(self, texts: list[str]):
        if any(not text.strip() for text in texts):
            raise ValueError("Text must not be empty.")
        return [self.analyze(text) for text in texts]


class FakeInferenceConfig:
    @classmethod
    def from_project(cls):
        return SimpleNamespace(
            sentiment_model_dir=SimpleNamespace(exists=lambda: True),
            topic_model_dir=SimpleNamespace(exists=lambda: True),
            toxic_baseline_path=SimpleNamespace(exists=lambda: True),
            urgency_baseline_path=SimpleNamespace(exists=lambda: True),
            project_dir="test-project",
        )


class FakeRetrievalService:
    def health(self):
        return {"status": "ready", "collection": "student_feedback", "points_count": 2}

    def search(self, query: str, top_k: int = 5, **_filters):
        if not query.strip():
            raise ValueError("Query must not be empty.")
        return [
            {
                "id": "feedback-1",
                "vector_score": 0.93,
                "rerank_score": 4.12,
                "text": "Wifi phong hoc qua yeu.",
                "source_dataset": "UIT_VSFC",
                "sentiment": "negative",
                "topic": "facilities",
                "toxic": 0,
                "urgency": "medium",
            }
        ][:top_k]

    def search_rankings(self, query: str, candidate_k: int = 20, **_filters):
        rows = self.search(query, top_k=candidate_k)
        return {
            "vector_results": [{**row, "vector_rank": index} for index, row in enumerate(rows, start=1)],
            "reranked_results": [{**row, "rerank_rank": index} for index, row in enumerate(rows, start=1)],
        }


class FakeRAGService:
    def __init__(self):
        self.received = None

    def ask(self, question: str, top_k: int = 6, **_filters):
        self.received = {"question": question, "top_k": top_k, **_filters}
        return {
            "question": question,
            "answer": "Wifi phong hoc yeu va can duoc cai thien. [1]",
            "evidence": [
                {
                    "rank": 1,
                    "id": "feedback-1",
                    "text": "Wifi phong hoc qua yeu.",
                    "topic": "facilities",
                    "rerank_score": 4.12,
                }
            ][:top_k],
            "retrieved_count": min(top_k, 1),
            "grounded": True,
        }


class FakeChatService:
    def __init__(self):
        self.sessions = {1: {"id": 1, "title": "Wifi", "created_at": "2026-01-01", "updated_at": "2026-01-01"}}

    def list_sessions(self, _limit):
        return list(self.sessions.values())

    def create_session(self, title):
        return {"id": 2, "title": title or "Cuoc tro chuyen moi", "created_at": "2026-01-01", "updated_at": "2026-01-01"}

    def get_session(self, session_id):
        session = self.sessions.get(session_id)
        return {**session, "messages": []} if session else None

    def delete_session(self, session_id):
        return self.sessions.pop(session_id, None) is not None

    def ask(self, session_id, question, **_filters):
        if session_id not in self.sessions:
            return None
        return {
            "session": self.sessions[session_id],
            "user_message": {"role": "user", "content": question},
            "assistant_message": {"role": "assistant", "content": "Tra loi [1]", "result": {"answer": "Tra loi [1]"}},
        }


class FakeAnalyticsService:
    def __init__(self):
        self.received = None

    def get_analytics(self, **filters):
        self.received = filters
        return {
            "total_feedback": 2,
            "filtered_feedback": 1 if filters.get("topic") else 2,
            "source_distribution": [{"label": "UIT_VSFC", "count": 2, "percentage": 100.0}],
            "sentiment_distribution": [{"label": "negative", "count": 2, "percentage": 100.0}],
            "topic_distribution": [{"label": "facilities", "count": 2, "percentage": 100.0}],
            "urgency_distribution": [{"label": "medium", "count": 2, "percentage": 100.0}],
            "toxic_distribution": [{"label": 0, "count": 2, "percentage": 100.0}],
            "negative_by_topic": [{"topic": "facilities", "count": 2}],
            "urgency_by_topic": [{"topic": "facilities", "urgency": "medium", "count": 2}],
            "sentiment_topic_matrix": [{"topic": "facilities", "sentiment": "negative", "count": 2}],
        }


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api_app, "InferenceConfig", FakeInferenceConfig)
    monkeypatch.setattr(api_app, "get_inference_service", lambda: FakeInferenceService())
    monkeypatch.setattr(api_app, "get_retrieval_service", lambda: FakeRetrievalService())
    monkeypatch.setattr(api_app, "get_rag_service", lambda: FakeRAGService())
    monkeypatch.setattr(api_app, "get_analytics_service", lambda: FakeAnalyticsService())
    monkeypatch.setattr(api_app, "get_chat_service", lambda: FakeChatService())
    return TestClient(api_app.app)


def test_root_returns_api_metadata(monkeypatch):
    response = make_client(monkeypatch).get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_does_not_require_model_loading(monkeypatch):
    response = make_client(monkeypatch).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "sentiment_model_exists": True,
        "topic_model_exists": True,
        "toxic_baseline_exists": True,
        "urgency_baseline_exists": True,
        "project_dir": "test-project",
    }


def test_analytics_returns_filtered_aggregations(monkeypatch):
    fake_analytics = FakeAnalyticsService()
    monkeypatch.setattr(api_app, "get_analytics_service", lambda: fake_analytics)

    response = TestClient(api_app.app).get("/analytics?topic=facilities&toxic=0")

    assert response.status_code == 200
    assert response.json()["filtered_feedback"] == 1
    assert fake_analytics.received == {
        "dataset": None,
        "topic": "facilities",
        "sentiment": None,
        "urgency": None,
        "toxic": 0,
    }


def test_predict_returns_expected_analysis(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict",
        json={"text": "Wifi phong hoc qua yeu."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "negative"
    assert body["topic"] == "facility"
    assert body["toxic"] == 0
    assert body["urgency"] == "medium"


def test_predict_rejects_empty_text(monkeypatch):
    response = make_client(monkeypatch).post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_batch_returns_one_result_per_text(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict-batch",
        json={"texts": ["Giảng viên dạy dễ hiểu.", "Wifi quá yếu."]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert [row["text"] for row in results] == [
        "Giảng viên dạy dễ hiểu.",
        "Wifi quá yếu.",
    ]


def test_predict_batch_rejects_blank_item(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict-batch",
        json={"texts": ["Noi dung hop le.", "  "]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Text must not be empty."


def test_predict_csv_returns_predictions_file(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict-csv",
        files={
            "file": (
                "feedback.csv",
                "student_id,text\nsv-01,Wifi phong hoc qua yeu.\nsv-02,Giang vien day de hieu.\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=student_voice_predictions.csv" in response.headers[
        "content-disposition"
    ]
    rows = response.content.decode("utf-8-sig").splitlines()
    assert rows[0] == (
        "student_id,text,sentiment,sentiment_confidence,topic,topic_confidence,toxic,urgency"
    )
    assert len(rows) == 3
    assert rows[1].startswith("sv-01,Wifi phong hoc qua yeu.,negative,0.91,facility,0.88,0,medium")


def test_predict_csv_rejects_missing_text_column(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict-csv",
        files={"file": ("feedback.csv", "student_id,feedback\nsv-01,WIFI yeu.\n", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "CSV must contain a 'text' column."


def test_predict_csv_rejects_file_without_rows(monkeypatch):
    response = make_client(monkeypatch).post(
        "/predict-csv",
        files={"file": ("feedback.csv", "text\n", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "CSV must contain at least one row."


def test_search_returns_similar_feedback(monkeypatch):
    response = make_client(monkeypatch).post(
        "/search",
        json={"query": "wifi phong hoc yeu", "top_k": 5, "topic": "facilities"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "wifi phong hoc yeu"
    assert body["results"][0]["topic"] == "facilities"
    assert body["results"][0]["vector_score"] == 0.93
    assert body["results"][0]["rerank_score"] == 4.12


def test_search_rejects_invalid_top_k(monkeypatch):
    response = make_client(monkeypatch).post(
        "/search",
        json={"query": "wifi", "top_k": 0},
    )

    assert response.status_code == 422


def test_search_compare_returns_both_rankings(monkeypatch):
    response = make_client(monkeypatch).post(
        "/search/compare",
        json={"query": "wifi phong hoc yeu", "candidate_k": 10, "topic": "facilities"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_k"] == 10
    assert body["vector_results"][0]["vector_rank"] == 1
    assert body["reranked_results"][0]["rerank_rank"] == 1


def test_ask_returns_grounded_answer_and_evidence(monkeypatch):
    response = make_client(monkeypatch).post(
        "/ask",
        json={"question": "Sinh vien noi gi ve wifi?", "top_k": 6, "topic": "facilities"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["answer"].endswith("[1]")
    assert body["evidence"][0]["topic"] == "facilities"


def test_ask_forwards_filters_to_rag_service(monkeypatch):
    fake_rag = FakeRAGService()
    monkeypatch.setattr(api_app, "get_rag_service", lambda: fake_rag)

    response = TestClient(api_app.app).post(
        "/ask",
        json={
            "question": "Phong hoc co van de gi?",
            "top_k": 3,
            "topic": "facilities",
            "sentiment": "negative",
            "urgency": "medium",
            "toxic": 0,
        },
    )

    assert response.status_code == 200
    assert fake_rag.received == {
        "question": "Phong hoc co van de gi?",
        "top_k": 3,
        "topic": "facilities",
        "sentiment": "negative",
        "urgency": "medium",
        "toxic": 0,
        "history": [],
    }


def test_chat_session_api_routes(monkeypatch):
    client = make_client(monkeypatch)

    listed = client.get("/chat-sessions")
    created = client.post("/chat-sessions", json={})
    opened = client.get("/chat-sessions/1")
    answer = client.post("/chat-sessions/1/ask", json={"question": "Wifi yeu?"})
    deleted = client.delete("/chat-sessions/1")

    assert listed.status_code == 200
    assert listed.json()["items"][0]["title"] == "Wifi"
    assert created.status_code == 200
    assert opened.status_code == 200
    assert answer.json()["assistant_message"]["result"]["answer"] == "Tra loi [1]"
    assert deleted.json() == {"deleted": True}
