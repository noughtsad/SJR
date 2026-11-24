import numpy as np

from app.recommend.explain import Gap, LLMExplanation, Reason, is_grounded, verify
from app.recommend.scoring import cap_per_company, match_signals, rocchio, seniority_gap


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# --- Rocchio ---------------------------------------------------------------


def test_rocchio_no_feedback_is_identity():
    base = _unit([1, 0, 0])
    assert np.allclose(rocchio(base, [], []), base)


def test_rocchio_moves_toward_saved_and_away_from_dismissed():
    base, liked, disliked = _unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])
    q = rocchio(base, [liked], [disliked])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert q @ liked > base @ liked
    assert q @ disliked < base @ disliked
    assert q @ base > q @ liked  # profile still dominates with default weights


def test_rocchio_uses_centroid_not_sum():
    base, liked = _unit([1, 0]), _unit([0, 1])
    one = rocchio(base, [liked], [])
    many = rocchio(base, [liked] * 10, [])
    assert np.allclose(one, many)  # ten saves of similar jobs != ten times the pull


# --- scoring ---------------------------------------------------------------


def test_match_signals_terms():
    m = match_signals(0.8, ["python", "sql", "azure"], ["python", "sql", "kafka"], "senior", "senior")
    assert m.matched_skills == ["python", "sql"] and m.missing_skills == ["azure"]
    assert np.isclose(m.skill_coverage, 2 / 3)
    assert np.isclose(m.score, 0.8 + 0.10 * 2 / 4)  # smoothed: 2 / (3 + 1)


def test_single_tag_job_does_not_beat_full_overlap():
    one = match_signals(0.8, ["python"], ["python", "sql", "aws", "dbt", "kafka"], None, None)
    five = match_signals(
        0.8, ["python", "sql", "aws", "dbt", "kafka"], ["python", "sql", "aws", "dbt", "kafka"], None, None
    )
    assert one.skill_coverage == five.skill_coverage == 1.0
    assert five.score > one.score


def test_seniority_gap():
    assert seniority_gap("senior", "junior") == 2
    assert seniority_gap(None, "senior") == 0
    m = match_signals(0.8, [], [], "intern", "senior")
    assert m.skill_coverage is None and np.isclose(m.score, 0.8 - 0.03 * 3)


def test_cap_per_company_keeps_order():
    items = [("a", "Acme"), ("b", "acme "), ("c", "Beta"), ("d", "ACME"), ("e", None), ("f", None)]
    out = cap_per_company(items, lambda x: x[1], 2)
    assert [x[0] for x in out] == ["a", "b", "c", "e", "f"]


# --- grounding -------------------------------------------------------------

JOB = 'Senior Data Engineer at Acme\n\nYou will build Kafka pipelines.  Experience with "Azure Data Factory" required.'
RESUME = "Built a Kafka + Spark pipeline processing 40M events/day. Python, SQL, Kafka."


def test_is_grounded_exact_and_normalised():
    assert is_grounded("build Kafka pipelines", JOB)
    assert is_grounded("experience with azure data factory", JOB)  # case + quotes ignored
    assert not is_grounded("build Flink pipelines", JOB)
    assert not is_grounded("Kafka", JOB)  # too short to count as evidence


def test_is_grounded_ellipsis_pieces_in_order():
    assert is_grounded("You will build ... Azure Data Factory", JOB)
    assert not is_grounded("Azure Data Factory ... You will build", JOB)


def test_verify_drops_ungrounded_and_contradictory_claims():
    expl = LLMExplanation(
        fit="strong",
        summary="ok",
        reasons=[
            Reason(
                point="streaming",
                job_evidence="build Kafka pipelines",
                profile_evidence="Built a Kafka + Spark pipeline",
            ),
            Reason(
                point="invented", job_evidence="lead a team of 10", profile_evidence="Built a Kafka + Spark pipeline"
            ),
            Reason(point="bad resume quote", job_evidence="build Kafka pipelines", profile_evidence="10 years of Rust"),
        ],
        gaps=[
            Gap(requirement="Azure Data Factory", job_evidence="Experience with Azure Data Factory required"),
            Gap(requirement="Kafka experience", job_evidence="build Kafka pipelines"),  # user has kafka
            Gap(requirement="Go", job_evidence="Go services in production"),  # not in posting
        ],
    )
    out, dropped = verify(expl, JOB, RESUME, ["kafka", "python", "sql"])
    assert [r.point for r in out.reasons] == ["streaming"]
    assert [g.requirement for g in out.gaps] == ["Azure Data Factory"]
    assert len(dropped) == 4
