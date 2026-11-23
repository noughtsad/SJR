import math

from app.evals.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k


def test_single_target():
    ranked = ["a", "b", "t", "c"]
    qrels = {"t": 2}
    assert recall_at_k(ranked, qrels, 10) == 1.0
    assert recall_at_k(ranked, qrels, 2) == 0.0
    assert mrr_at_k(ranked, qrels) == 1 / 3
    assert math.isclose(ndcg_at_k(ranked, qrels), 1 / math.log2(4))


def test_graded_ndcg_prefers_highly_relevant_first():
    qrels = {"x": 2, "y": 1}
    assert ndcg_at_k(["x", "y"], qrels) == 1.0
    assert ndcg_at_k(["y", "x"], qrels) < 1.0


def test_missing_relevant_doc_is_penalised():
    qrels = {"x": 2, "y": 2}
    assert recall_at_k(["x", "z"], qrels) == 0.5
    assert ndcg_at_k(["x", "z"], qrels) < 1.0


def test_no_relevant_and_zero_grades():
    assert mrr_at_k(["a"], {"a": 0}) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0
    assert recall_at_k(["a"], {"a": 0}) == 0.0


def test_percentile():
    v = list(range(1, 101))
    assert percentile(v, 50) == 50
    assert percentile(v, 95) == 95
    assert percentile([7], 95) == 7
