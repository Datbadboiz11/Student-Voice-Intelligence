from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


REFUSAL = "không đủ dữ liệu để kết luận"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an AI-assisted preliminary RAG quality evaluation.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/rag_quality_queries.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/evaluation"))
    return parser.parse_args()


def payload(case: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"question": case["question"], "top_k": 6}
    for field in ("topic", "sentiment", "urgency"):
        if case.get(field, "").strip():
            result[field] = case[field].strip()
    return result


def groups_score(answer: str, groups: str) -> int:
    expected = [group.split("|") for group in groups.split(";") if group]
    if not expected:
        return 5
    normalized = unicodedata.normalize("NFD", answer.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")
    def contains(term: str) -> bool:
        value = unicodedata.normalize("NFD", term.lower())
        value = "".join(char for char in value if unicodedata.category(char) != "Mn").replace("đ", "d")
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", normalized))
    matched = sum(any(contains(term) for term in group) for group in expected)
    return max(1, round(5 * matched / len(expected)))


def citation_score(answer: str, evidence_count: int) -> int:
    citations = [int(value) for value in re.findall(r"\[(\d+)]", answer)]
    if not citations:
        return 1
    valid = sum(1 <= value <= evidence_count for value in citations)
    return max(1, round(5 * valid / len(citations)))


def clarity_score(answer: str) -> int:
    bullets = [line for line in answer.splitlines() if line.strip().startswith(("-", "•"))]
    if len(answer) > 1500 or len(bullets) > 4:
        return 3
    if len(answer) < 30:
        return 2
    return 5


def score(case: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer", ""))
    refusal = REFUSAL in answer.lower()
    expected_refusal = case["expected_behavior"] == "insufficient_data"
    evidence = result.get("evidence", [])
    if expected_refusal:
        value = 5 if refusal else 1
        return {"relevance": value, "faithfulness": value, "coverage": value, "citation_quality": 5 if refusal else 1, "clarity": 5 if refusal else clarity_score(answer)}
    coverage = groups_score(answer, case.get("expected_groups", ""))
    citations = citation_score(answer, len(evidence))
    return {
        "relevance": coverage,
        "faithfulness": citations if result.get("grounded") else 1,
        "coverage": coverage,
        "citation_quality": citations,
        "clarity": clarity_score(answer),
    }


def main() -> None:
    args = parse_args()
    with args.cases.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for case in cases:
            response = client.post(f"{args.api_url.rstrip('/')}/ask", json=payload(case))
            response.raise_for_status()
            result = response.json()
            scores = score(case, result)
            rows.append({**case, **scores, "answer": result.get("answer", ""), "evidence_count": result.get("retrieved_count", 0), "grounded": result.get("grounded", False), "label_source": "AI-assisted preliminary"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rag_quality_evaluation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("relevance", "faithfulness", "coverage", "citation_quality", "clarity")
    report = ["# RAG Quality Evaluation", "", "Nhãn là AI-assisted preliminary; cần audit con người trước khi dùng làm kết quả chính thức.", "", "| Tiêu chí | Điểm trung bình / 5 |", "|---|---:|"]
    report.extend(f"| {metric} | {mean(float(row[metric]) for row in rows):.2f} |" for metric in metrics)
    report.extend(["", f"Số câu hỏi: **{len(rows)}**."])
    (args.output_dir / "rag_evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {csv_path} and {args.output_dir / 'rag_evaluation.md'}.")


if __name__ == "__main__":
    main()
