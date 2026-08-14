from __future__ import annotations

from evals.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_and_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}  # z isn't retrievable at all -> caps recall below 1.0

    assert precision_at_k(ranked, relevant, k=4) == 0.5  # 2 of 4 retrieved are relevant
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3  # 2 of 3 relevant were found


def test_mrr_is_reciprocal_rank_of_first_hit() -> None:
    assert mrr(["a", "b", "c"], {"c"}, k=3) == 1 / 3
    assert mrr(["a", "b", "c"], {"a"}, k=3) == 1.0
    assert mrr(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_ndcg_rewards_relevant_items_ranked_higher() -> None:
    relevant = {"x"}
    # Same single hit, earlier rank must score higher (that's the whole point of NDCG
    # over plain precision, which can't tell rank 1 from rank 3 apart).
    first = ndcg_at_k(["x", "b", "c"], relevant, k=3)
    last = ndcg_at_k(["a", "b", "x"], relevant, k=3)
    assert first == 1.0  # perfect: the only relevant item is ranked first
    assert 0 < last < first


def test_metrics_are_zero_with_no_relevant_items_in_range() -> None:
    ranked = ["a", "b"]
    relevant = {"z"}
    assert precision_at_k(ranked, relevant, k=2) == 0.0
    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert mrr(ranked, relevant, k=2) == 0.0
    assert ndcg_at_k(ranked, relevant, k=2) == 0.0
