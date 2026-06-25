from scripts.evaluate_retrieval import ranking_metrics


def test_rerank_metrics_reward_earlier_relevant_feedback():
    rows = [
        {"feedback_id": "a", "relevance": 0, "rank_vector": 1, "rank_rerank": 2},
        {"feedback_id": "b", "relevance": 1, "rank_vector": 2, "rank_rerank": 3},
        {"feedback_id": "c", "relevance": 2, "rank_vector": 3, "rank_rerank": 1},
    ]

    vector = ranking_metrics(rows, "rank_vector")
    reranked = ranking_metrics(rows, "rank_rerank")

    assert vector["Recall@5"] == 1.0
    assert reranked["Recall@5"] == 1.0
    assert reranked["MRR"] > vector["MRR"]
    assert reranked["nDCG@5"] > vector["nDCG@5"]
