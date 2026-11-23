"""Stage 3: Reciprocal Rank Fusion.

    rrf(d) = sum over retrievers r of  w_r / (k + rank_r(d))

Only ranks are used, never raw scores. BM25-style ts_rank and cosine
similarity are on incomparable scales, so mixing the scores directly would need
calibration; ranks don't. k (default 60, from Cormack et al. 2009) flattens the
curve so one retriever's #1 can't dominate a document that both rank highly.
"""

from app.search.models import Candidate


def rrf_fuse(
    keyword: list[Candidate],
    vector: list[Candidate],
    k: int = 60,
    w_keyword: float = 1.0,
    w_vector: float = 1.0,
) -> list[Candidate]:
    merged: dict[int, Candidate] = {}
    for c in keyword:
        merged[c.job_id] = c.model_copy()
    for c in vector:
        m = merged.setdefault(c.job_id, Candidate(job_id=c.job_id))
        m.vector_rank, m.vector_score = c.vector_rank, c.vector_score

    for m in merged.values():
        s = 0.0
        if m.keyword_rank is not None:
            s += w_keyword / (k + m.keyword_rank)
        if m.vector_rank is not None:
            s += w_vector / (k + m.vector_rank)
        m.rrf_score = s

    # tie-break: prefer docs found by both retrievers, then better best-rank
    def key(m: Candidate):
        best = min(r for r in (m.keyword_rank, m.vector_rank) if r is not None)
        both = m.keyword_rank is not None and m.vector_rank is not None
        return (-m.rrf_score, not both, best)

    return sorted(merged.values(), key=key)
