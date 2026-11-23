"""Ranking metrics over one query.

ranked:  the system's result ids, best first
qrels:   {doc_id: graded relevance}; 0/absent = not relevant, >=1 relevant, 2 = highly relevant
"""

import math


def recall_at_k(ranked: list, qrels: dict, k: int = 10) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list, qrels: dict, k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list, qrels: dict, k: int = 10) -> float:
    """Exponential gain (2^rel - 1) so 'highly relevant' outweighs 'partially'.
    The ideal ranking uses *all* judged-relevant docs, not just retrieved ones,
    so missing a relevant doc is penalised."""

    def dcg(gains):
        return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains, 1))

    ideal = dcg(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(d, 0) for d in ranked[:k]]) / ideal


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, no numpy needed."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
