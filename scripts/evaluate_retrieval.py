from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("Recall@5", "Recall@10", "MRR", "nDCG@5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vector retrieval against CrossEncoder reranking.")
    parser.add_argument("--judgments", type=Path, default=Path("data/evaluation/retrieval_judgments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/evaluation"))
    return parser.parse_args()


def dcg(relevances: list[int]) -> float:
    return sum((2**relevance - 1) / math.log2(index + 2) for index, relevance in enumerate(relevances))


def ranking_metrics(rows: list[dict[str, Any]], rank_key: str) -> dict[str, float]:
    ordered = sorted((row for row in rows if row[rank_key] is not None), key=lambda row: row[rank_key])
    relevant = [row for row in ordered if row["relevance"] > 0]
    if not relevant:
        raise ValueError("Every query must have at least one relevant feedback.")
    relevant_ids = {row["feedback_id"] for row in relevant}
    top5 = ordered[:5]
    top10 = ordered[:10]
    first_rank = next((index for index, row in enumerate(ordered, start=1) if row["relevance"] > 0), None)
    ideal = sorted((row["relevance"] for row in rows), reverse=True)[:5]
    return {
        "Recall@5": len({row["feedback_id"] for row in top5 if row["relevance"] > 0}) / len(relevant_ids),
        "Recall@10": len({row["feedback_id"] for row in top10 if row["relevance"] > 0}) / len(relevant_ids),
        "MRR": 1 / first_rank if first_rank else 0.0,
        "nDCG@5": dcg([row["relevance"] for row in top5]) / dcg(ideal) if dcg(ideal) else 0.0,
    }


def mean_metrics(rows: list[dict[str, Any]], rank_key: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    per_query = []
    for query_id, group in groups.items():
        metrics = ranking_metrics(group, rank_key)
        per_query.append({"query_id": query_id, "query": group[0]["query"], **metrics})
    return ({metric: sum(row[metric] for row in per_query) / len(per_query) for metric in METRICS}, per_query)


def parse_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("Judgment file is empty.")
    rows = []
    incomplete = []
    for row in raw_rows:
        value = row.get("relevance", "").strip()
        if value not in {"0", "1", "2"}:
            incomplete.append(f"{row.get('query_id')}:{row.get('feedback_id')}")
            continue
        rows.append(
            {
                **row,
                "relevance": int(value),
                "rank_vector": int(row["rank_vector"]) if row.get("rank_vector") else None,
                "rank_rerank": int(row["rank_rerank"]) if row.get("rank_rerank") else None,
            }
        )
    if incomplete:
        raise ValueError(f"{len(incomplete)} candidates have no relevance label. Complete every row before evaluation.")
    return rows


def main() -> None:
    args = parse_args()
    if not args.judgments.exists():
        raise FileNotFoundError(f"Judgment file not found: {args.judgments}")
    rows = parse_rows(args.judgments)
    vector_metrics, vector_queries = mean_metrics(rows, "rank_vector")
    rerank_metrics, rerank_queries = mean_metrics(rows, "rank_rerank")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "retrieval_metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *METRICS])
        writer.writeheader()
        writer.writerow({"method": "vector_only", **{key: f"{value:.4f}" for key, value in vector_metrics.items()}})
        writer.writerow({"method": "vector_plus_rerank", **{key: f"{value:.4f}" for key, value in rerank_metrics.items()}})
    detail_path = args.output_dir / "retrieval_per_query.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "query_id", "query", *METRICS])
        writer.writeheader()
        for method, per_query in (("vector_only", vector_queries), ("vector_plus_rerank", rerank_queries)):
            writer.writerows({"method": method, **row} for row in per_query)
    report_path = args.output_dir / "retrieval_evaluation.md"
    ai_assisted = any("AI-assisted" in row.get("review_note", "") for row in rows)
    label_source = (
        "Nhãn relevance là nhãn sơ bộ có hỗ trợ AI, cần được con người audit trước khi dùng làm kết quả chính thức."
        if ai_assisted
        else "Nhãn relevance được gán thủ công: 0 = không liên quan, 1 = liên quan, 2 = rất liên quan."
    )
    lines = [
        "# Retrieval và Rerank Evaluation",
        "",
        f"Số query được đánh giá: **{len(vector_queries)}**.",
        "",
        "| Phương pháp | Recall@5 | Recall@10 | MRR | nDCG@5 |",
        "|---|---:|---:|---:|---:|",
        "| Vector only | " + " | ".join(f"{vector_metrics[key]:.4f}" for key in METRICS) + " |",
        "| Vector + CrossEncoder rerank | " + " | ".join(f"{rerank_metrics[key]:.4f}" for key in METRICS) + " |",
        "",
        label_source,
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {metrics_path}, {detail_path}, and {report_path}.")


if __name__ == "__main__":
    main()
