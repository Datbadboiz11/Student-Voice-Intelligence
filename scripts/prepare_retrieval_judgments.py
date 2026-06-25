from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manual relevance-judgment sheet for retrieval evaluation.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--queries", type=Path, default=Path("data/evaluation/retrieval_queries.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/retrieval_judgments.csv"))
    parser.add_argument("--candidate-k", type=int, default=20, choices=range(10, 51))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_payload(query: dict[str, str], candidate_k: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query["query"], "candidate_k": candidate_k}
    for field in ("topic", "sentiment", "urgency", "toxic"):
        value = query.get(field, "").strip()
        if value:
            payload[field] = int(value) if field == "toxic" else value
    return payload


def main() -> None:
    args = parse_args()
    if not args.queries.exists():
        raise FileNotFoundError(f"Query set not found: {args.queries}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Use --overwrite only before manual labeling.")
    with args.queries.open(encoding="utf-8-sig", newline="") as handle:
        queries = list(csv.DictReader(handle))
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for item in queries:
            response = client.post(
                f"{args.api_url.rstrip('/')}/search/compare",
                json=request_payload(item, args.candidate_k),
            )
            response.raise_for_status()
            data = response.json()
            candidates: dict[str, dict[str, Any]] = {}
            for result in data["vector_results"]:
                candidates[result["id"]] = {**result, "rank_vector": result["vector_rank"], "rank_rerank": ""}
            for result in data["reranked_results"]:
                row = candidates.setdefault(result["id"], dict(result))
                row["rank_rerank"] = result["rerank_rank"]
                row.setdefault("rank_vector", "")
            for candidate in sorted(candidates.values(), key=lambda row: min(int(row.get("rank_vector") or 999), int(row.get("rank_rerank") or 999))):
                rows.append(
                    {
                        "query_id": item["query_id"],
                        "query": item["query"],
                        "feedback_id": candidate["id"],
                        "rank_vector": candidate["rank_vector"],
                        "rank_rerank": candidate["rank_rerank"],
                        "vector_score": candidate.get("vector_score", ""),
                        "rerank_score": candidate.get("rerank_score", ""),
                        "topic": candidate.get("topic", ""),
                        "sentiment": candidate.get("sentiment", ""),
                        "urgency": candidate.get("urgency", ""),
                        "text": candidate.get("text", ""),
                        "relevance": "",
                        "review_note": "",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} candidate judgments for {len(queries)} queries to {args.output}.")
    print("Set relevance to 0 (not relevant), 1 (relevant), or 2 (highly relevant), then run evaluate_retrieval.py.")


if __name__ == "__main__":
    main()
