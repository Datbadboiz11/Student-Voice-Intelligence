from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import httpx


INSUFFICIENT_DATA = "không đủ dữ liệu để kết luận"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Student Voice RAG against a CSV test set.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/evaluation/rag_test_cases.csv"),
        help="Input CSV containing RAG test cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/rag_evaluation.csv"),
        help="Output CSV with test results",
    )
    return parser.parse_args()


def evaluate_case(client: httpx.Client, api_url: str, case: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": case["question"], "top_k": 6}
    for case_field, api_field in (
        ("expected_topic", "topic"),
        ("expected_sentiment", "sentiment"),
        ("expected_urgency", "urgency"),
        ("expected_toxic", "toxic"),
    ):
        value = case.get(case_field, "").strip()
        if value:
            payload[api_field] = int(value) if api_field == "toxic" else value
    response = client.post(f"{api_url.rstrip('/')}/ask", json=payload)
    response.raise_for_status()
    result = response.json()
    answer = str(result.get("answer", ""))
    answer_lower = answer.lower()
    expected_behavior = case["expected_behavior"]
    refused = INSUFFICIENT_DATA in answer_lower
    behavior_correct = refused if expected_behavior == "insufficient_data" else not refused
    evidence = result.get("evidence", [])
    topic_correct = not case.get("expected_topic") or any(
        item.get("topic") == case["expected_topic"] for item in evidence
    )
    keywords = [item.strip().lower() for item in case.get("expected_keywords", "").split("|") if item.strip()]
    keyword_score = 1.0 if not keywords else sum(keyword in answer_lower for keyword in keywords) / len(keywords)
    return {
        **case,
        "status": "ok",
        "behavior_correct": behavior_correct,
        "topic_correct": topic_correct,
        "keyword_score": round(keyword_score, 2),
        "retrieved_count": result.get("retrieved_count", 0),
        "grounded": result.get("grounded", False),
        "answer": answer,
    }


def main() -> None:
    args = parse_args()
    if not args.cases.exists():
        raise FileNotFoundError(f"Test cases not found: {args.cases}")
    with args.cases.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    rows = []
    with httpx.Client(timeout=90.0) as client:
        for case in cases:
            try:
                rows.append(evaluate_case(client, args.api_url, case))
            except httpx.HTTPError as exc:
                rows.append({**case, "status": "error", "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row.get("behavior_correct") is True and row.get("topic_correct") is True for row in rows)
    print(f"Saved {len(rows)} cases to {args.output}. Automatic checks passed: {passed}/{len(rows)}.")


if __name__ == "__main__":
    main()
