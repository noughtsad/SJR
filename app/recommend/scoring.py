"""Pure functions for recommendations (no I/O, unit-tested).

Rocchio relevance feedback (Rocchio 1971; Manning et al., IIR ch. 9):
    q' = a * q0 + b * mean(saved) - c * mean(dismissed)
q0 is the resume-profile vector. Saving jobs pulls the query toward them,
dismissing pushes it away (weaker, because "not this one" is a noisier signal
than "more like this"). The result is re-normalised for cosine search.
"""

from dataclasses import dataclass

import numpy as np

from app.ingestion.normalize import SENIORITY_LEVELS


def rocchio(
    base: np.ndarray,
    saved: list[np.ndarray],
    dismissed: list[np.ndarray],
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    q = alpha * np.asarray(base, dtype=np.float32)
    if saved:
        q = q + beta * np.mean(saved, axis=0)
    if dismissed:
        q = q - gamma * np.mean(dismissed, axis=0)
    n = float(np.linalg.norm(q))
    return (q / n).astype(np.float32) if n > 0 else np.asarray(base, dtype=np.float32)


@dataclass
class MatchSignals:
    similarity: float
    matched_skills: list[str]
    missing_skills: list[str]
    skill_coverage: float | None  # share of the job's tagged skills the user has; None if job has no tags
    seniority_gap: int  # |levels apart|, 0 if either side is unknown
    score: float


def seniority_gap(job_level: str | None, user_level: str | None) -> int:
    if job_level not in SENIORITY_LEVELS or user_level not in SENIORITY_LEVELS:
        return 0
    return abs(SENIORITY_LEVELS.index(job_level) - SENIORITY_LEVELS.index(user_level))


def match_signals(
    similarity: float,
    job_skills: list[str],
    user_skills: list[str],
    job_level: str | None,
    user_level: str | None,
    skill_weight: float = 0.10,
    seniority_penalty: float = 0.03,
) -> MatchSignals:
    """Final score = cosine + skill bonus - seniority penalty. The extra terms are
    small on purpose: they re-order near-ties among semantically similar jobs,
    they don't override the embedding. Each term is returned so the UI can show
    exactly why a job ranked where it did."""
    js, us = set(job_skills), set(user_skills)
    matched, missing = sorted(js & us), sorted(js - us)
    coverage = len(matched) / len(js) if js else None
    # The score uses add-one smoothing: a job tagged with a single skill the user has
    # (1/1) shouldn't outrank one where they have 5 of 5. 1/1 -> 0.5, 5/5 -> 0.83.
    smoothed = len(matched) / (len(js) + 1)
    gap = seniority_gap(job_level, user_level)
    score = similarity + skill_weight * smoothed - seniority_penalty * gap
    return MatchSignals(similarity, matched, missing, coverage, gap, score)


def cap_per_company[T](items: list[T], company_of, max_per_company: int = 2) -> list[T]:
    """Keep ranking order but allow at most N results per company. Staffing agencies
    post the same role many times with small edits; without a cap they fill a page."""
    counts: dict[str, int] = {}
    out = []
    for it in items:
        c = (company_of(it) or "").strip().lower()
        if c:
            if counts.get(c, 0) >= max_per_company:
                continue
            counts[c] = counts.get(c, 0) + 1
        out.append(it)
    return out
