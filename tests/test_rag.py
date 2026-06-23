from src.rag import OpenAIGenerator, RAGService, get_llm_generator


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.received = None

    def search(self, **kwargs):
        self.received = kwargs
        return self.results


class FakeGenerator:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "Sinh vien phan anh Wifi yeu o phong hoc. [1]"


def test_rag_returns_grounded_answer_with_evidence():
    retriever = FakeRetriever(
        [
            {
                "id": "feedback-1",
                "text": "Wifi phong hoc qua yeu, khong the hoc on dinh.",
                "topic": "facilities",
                "sentiment": "negative",
                "urgency": "medium",
                "rerank_score": 4.12,
                "vector_score": 0.93,
            }
        ]
    )
    generator = FakeGenerator()

    result = RAGService(retriever=retriever, generator=generator).ask(
        "Sinh vien noi gi ve Wifi?", top_k=4, topic="facilities"
    )

    assert result["grounded"] is True
    assert result["retrieved_count"] == 1
    assert result["evidence"][0]["rank"] == 1
    assert "[1]" in generator.prompt
    assert "tối đa bốn gạch đầu dòng" in generator.prompt
    assert retriever.received["topic"] == "facilities"


def test_rag_does_not_call_generator_when_no_feedback_matches():
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    result = RAGService(retriever=retriever, generator=generator).ask("Cau hoi khong co du lieu")

    assert result["grounded"] is False
    assert result["evidence"] == []
    assert "Không đủ dữ liệu" in result["answer"]
    assert generator.prompt is None


def test_openai_generator_uses_responses_api(monkeypatch):
    requests = []

    class FakeResponse:
        is_error = False

        def json(self):
            return {"output": [{"content": [{"type": "output_text", "text": "Câu trả lời [1]"}]}]}

    def fake_post(*args, **kwargs):
        requests.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.rag.httpx.post", fake_post)

    answer = OpenAIGenerator("gpt-4o-mini").generate("prompt")

    assert answer == "Câu trả lời [1]"
    assert requests[0][0][0] == "https://api.openai.com/v1/responses"
    assert requests[0][1]["json"]["model"] == "gpt-4o-mini"


def test_llm_provider_selects_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    assert isinstance(get_llm_generator(), OpenAIGenerator)
