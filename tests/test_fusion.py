from app.search.fusion import rrf_fuse
from app.search.models import Candidate


def kw(*ids):
    return [Candidate(job_id=j, keyword_rank=i, keyword_score=1.0 / i) for i, j in enumerate(ids, 1)]


def vec(*ids):
    return [Candidate(job_id=j, vector_rank=i, vector_score=1.0 - i / 100) for i, j in enumerate(ids, 1)]


def test_doc_in_both_lists_beats_single_list_top():
    out = rrf_fuse(kw(1, 2, 3), vec(4, 2, 5), k=60)
    assert out[0].job_id == 2  # rank 2 in both > rank 1 in one
    assert out[0].keyword_rank == 2 and out[0].vector_rank == 2


def test_scores_match_formula():
    out = {c.job_id: c for c in rrf_fuse(kw(1, 2), vec(2, 3), k=60)}
    assert abs(out[2].rrf_score - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(out[1].rrf_score - 1 / 61) < 1e-12
    assert out[3].keyword_rank is None


def test_union_and_no_input_mutation():
    a, b = kw(1, 2), vec(3)
    out = rrf_fuse(a, b)
    assert {c.job_id for c in out} == {1, 2, 3}
    assert a[0].rrf_score is None


def test_empty_inputs():
    assert rrf_fuse([], []) == []
    assert [c.job_id for c in rrf_fuse([], vec(7, 8))] == [7, 8]
