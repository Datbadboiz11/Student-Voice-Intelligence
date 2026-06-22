from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

TOXIC_KEYWORDS = [
    "chửi",
    "thô tục",
    "xúc phạm",
    "miệt thị",
    "đe dọa",
    "bạo lực",
    "quấy rối",
    "lăng mạ",
    "ngu",
    "óc",
    "đồ điên",
    "biến thái",
]

HIGH_URGENCY_KEYWORDS = [
    "khẩn cấp",
    "ngay lập tức",
    "đe dọa",
    "bạo lực",
    "quấy rối",
    "nguy hiểm",
    "tai nạn",
    "cấp cứu",
    "an toàn",
    "cháy",
    "trộm",
]

MEDIUM_URGENCY_KEYWORDS = [
    "không thể",
    "bị lỗi",
    "quá chậm",
    "phản hồi chậm",
    "mất",
    "hỏng",
    "quá nóng",
    "quá yếu",
    "khó học",
    "khó chịu",
    "cần xử lý",
]

URGENCY_RANK = {"low": 0, "medium": 1, "high": 2}
URGENCY_BY_RANK = {rank: label for label, rank in URGENCY_RANK.items()}


def find_project_dir(start: Path | None = None) -> Path:
    env_project_dir = os.getenv("STUDENT_VOICE_PROJECT_DIR")
    if env_project_dir:
        return Path(env_project_dir).expanduser().resolve()

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find project root. Set STUDENT_VOICE_PROJECT_DIR to the project path."
    )


def normalize_text(text: str) -> str:
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=1)
def get_underthesea_word_tokenize() -> Any | None:
    try:
        from underthesea import word_tokenize

        return word_tokenize
    except Exception:
        return None


def has_underthesea() -> bool:
    return get_underthesea_word_tokenize() is not None


def segment_for_phobert(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    word_tokenize = get_underthesea_word_tokenize()
    if word_tokenize is not None:
        return word_tokenize(text, format="text")
    return text


def keyword_contains(text: str, keywords: list[str]) -> bool:
    text_lower = normalize_text(text).lower()
    return any(keyword in text_lower for keyword in keywords)


def predict_toxic_rule(text: str) -> int:
    return int(keyword_contains(text, TOXIC_KEYWORDS))


def predict_urgency_rule(text: str) -> str:
    if keyword_contains(text, HIGH_URGENCY_KEYWORDS):
        return "high"
    if keyword_contains(text, MEDIUM_URGENCY_KEYWORDS):
        return "medium"
    return "low"


def normalize_urgency_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label in URGENCY_RANK:
        return label
    return "low"


def merge_urgency_labels(baseline_label: str, rule_label: str) -> str:
    baseline_label = normalize_urgency_label(baseline_label)
    rule_label = normalize_urgency_label(rule_label)
    final_rank = max(URGENCY_RANK[baseline_label], URGENCY_RANK[rule_label])
    return URGENCY_BY_RANK[final_rank]


def check_model_dir(model_dir: Path, name: str) -> None:
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.txt",
        "bpe.codes",
    ]
    missing = [file_name for file_name in required_files if not (model_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError(f"{name} missing files {missing} in {model_dir}")


@dataclass(frozen=True)
class InferenceConfig:
    project_dir: Path
    sentiment_model_dir: Path
    topic_model_dir: Path
    toxic_baseline_path: Path
    urgency_baseline_path: Path
    max_length: int = 192

    @classmethod
    def from_project(cls, project_dir: Path | None = None) -> "InferenceConfig":
        root = project_dir or find_project_dir()
        return cls(
            project_dir=root,
            sentiment_model_dir=root / "outputs/models/transformer/phobertv2_sentiment",
            topic_model_dir=root / "outputs/models/transformer/phobertv2_topic_noweight",
            toxic_baseline_path=root / "outputs/models/baseline/toxic_binary_tfidf_linear_svm.joblib",
            urgency_baseline_path=root / "outputs/models/baseline/urgency_final_tfidf_linear_svm.joblib",
        )


class StudentVoiceInference:

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig.from_project()
        check_model_dir(self.config.sentiment_model_dir, "Sentiment model")
        check_model_dir(self.config.topic_model_dir, "Topic model")

        import joblib
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            self.config.sentiment_model_dir,
            local_files_only=True,
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.sentiment_model_dir,
            local_files_only=True,
        ).to(self.device)
        self.sentiment_model.eval()

        self.topic_tokenizer = AutoTokenizer.from_pretrained(
            self.config.topic_model_dir,
            local_files_only=True,
        )
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.topic_model_dir,
            local_files_only=True,
        ).to(self.device)
        self.topic_model.eval()

        self.toxic_model = (
            joblib.load(self.config.toxic_baseline_path)
            if self.config.toxic_baseline_path.exists()
            else None
        )
        self.urgency_model = (
            joblib.load(self.config.urgency_baseline_path)
            if self.config.urgency_baseline_path.exists()
            else None
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": str(self.device),
            "has_underthesea": has_underthesea(),
            "has_toxic_baseline": self.toxic_model is not None,
            "has_urgency_baseline": self.urgency_model is not None,
            "sentiment_labels": self.sentiment_model.config.id2label,
            "topic_labels": self.topic_model.config.id2label,
        }

    def _predict_transformer(
        self,
        texts: list[str],
        tokenizer: Any,
        model: Any,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        rows: list[dict[str, Any]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            phobert_texts = [segment_for_phobert(text) for text in batch_texts]

            encoded = tokenizer(
                phobert_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            import numpy as np

            with self.torch.no_grad():
                logits = model(**encoded).logits
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for original_text, phobert_text, prob in zip(batch_texts, phobert_texts, probs):
                pred_id = int(np.argmax(prob))
                probabilities = {
                    label: round(float(value), 6) for label, value in zip(labels, prob)
                }
                rows.append(
                    {
                        "text": original_text,
                        "text_phobert": phobert_text,
                        "label": labels[pred_id],
                        "confidence": round(float(prob[pred_id]), 6),
                        "probabilities": probabilities,
                    }
                )

        return rows

    def _predict_toxic(self, texts: list[str]) -> list[int]:
        rule_preds = [predict_toxic_rule(text) for text in texts]
        if self.toxic_model is None:
            return rule_preds

        baseline_preds = [int(value) for value in self.toxic_model.predict(texts)]
        return [int(bool(base) or bool(rule)) for base, rule in zip(baseline_preds, rule_preds)]

    def _predict_urgency(self, texts: list[str]) -> list[str]:
        rule_preds = [predict_urgency_rule(text) for text in texts]
        if self.urgency_model is None:
            return rule_preds

        baseline_preds = [normalize_urgency_label(value) for value in self.urgency_model.predict(texts)]
        return [
            merge_urgency_labels(base, rule)
            for base, rule in zip(baseline_preds, rule_preds)
        ]

    def analyze_many(self, texts: list[str]) -> list[dict[str, Any]]:
        cleaned_texts = [normalize_text(text) for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("Text must not be empty.")

        sentiment_rows = self._predict_transformer(
            cleaned_texts,
            self.sentiment_tokenizer,
            self.sentiment_model,
        )
        topic_rows = self._predict_transformer(
            cleaned_texts,
            self.topic_tokenizer,
            self.topic_model,
        )
        toxic_preds = self._predict_toxic(cleaned_texts)
        urgency_preds = self._predict_urgency(cleaned_texts)

        results: list[dict[str, Any]] = []
        for text, sentiment, topic, toxic, urgency in zip(
            cleaned_texts,
            sentiment_rows,
            topic_rows,
            toxic_preds,
            urgency_preds,
        ):
            results.append(
                {
                    "text": text,
                    "sentiment": sentiment["label"],
                    "sentiment_confidence": sentiment["confidence"],
                    "sentiment_probabilities": sentiment["probabilities"],
                    "topic": topic["label"],
                    "topic_confidence": topic["confidence"],
                    "topic_probabilities": topic["probabilities"],
                    "toxic": toxic,
                    "urgency": urgency,
                }
            )

        return results

    def analyze(self, text: str) -> dict[str, Any]:
        return self.analyze_many([text])[0]


@lru_cache(maxsize=1)
def get_inference_service() -> StudentVoiceInference:
    return StudentVoiceInference()
