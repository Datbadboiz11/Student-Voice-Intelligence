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


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api_app, "InferenceConfig", FakeInferenceConfig)
    monkeypatch.setattr(api_app, "get_inference_service", lambda: FakeInferenceService())
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
