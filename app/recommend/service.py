"""Profile + feedback -> ranked jobs.

1. Embed the profile (written like a job posting) as a document vector.
2. Rocchio-adjust it with the vectors of jobs the user saved / dismissed.
3. ANN search (same HNSW index as /search), excluding jobs already acted on.
4. Re-score the candidates with skill overlap and seniority fit.
"""

import asyncio
import time
from datetime import datetime

import numpy as np
from psycopg import sql
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.embeddings.registry import get_embedder, get_spec
from app.profile.extract import profile_text
from app.profile.store import get_feedback, get_profile
from app.recommend.scoring import cap_per_company, match_signals, rocchio
from app.search.models import SearchFilters
from app.search.service import hydrate
from app.search.vector import vector_search


class RecommendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    k: int = Field(10, ge=1, le=50)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_feedback: bool = True


class Recommendation(BaseModel):
    job_id: int
    title: str
    company: str | None
    location: str | None
    is_remote: bool
    seniority: str | None
    url: str | None
    posted_at: datetime | None
    snippet: str
    score: float
    similarity: float
    skill_coverage: float | None
    matched_skills: list[str]
    missing_skills: list[str]
    seniority_gap: int


class RecommendResponse(BaseModel):
    user_id: str
    profile_version: int
    feedback_used: dict[str, int]  # {"save": n, "dismiss": m}
    query_shift: float = Field(description="1 - cos(profile vector, feedback-adjusted vector)")
    results: list[Recommendation]
    timings_ms: dict[str, float]


async def _job_vectors(pool: AsyncConnectionPool, model_key: str, dim: int, ids: list[int]) -> dict[int, np.ndarray]:
    if not ids:
        return {}
    q = sql.SQL(
        "SELECT job_id, embedding::vector({dim}) AS v FROM job_embeddings WHERE model = {m} AND job_id = ANY(%s)"
    ).format(dim=sql.Literal(dim), m=sql.Literal(model_key))
    async with pool.connection() as conn:
        cur = await conn.execute(q, (ids,))
        return {r["job_id"]: np.asarray(r["v"].to_numpy(), dtype=np.float32) for r in await cur.fetchall()}


async def recommend(pool: AsyncConnectionPool, req: RecommendRequest) -> RecommendResponse:
    s = get_settings()
    ms: dict[str, float] = {}
    spec = get_spec()
    embedder = await asyncio.to_thread(get_embedder, spec.key)  # loads once; kept out of the timings
    t0 = time.perf_counter()

    stored, feedback = await asyncio.gather(get_profile(pool, req.user_id), get_feedback(pool, req.user_id))

    t = time.perf_counter()
    base = (await embedder.embed_documents([profile_text(stored.profile)]))[0]
    ms["embed_profile"] = round((time.perf_counter() - t) * 1000, 2)

    saved_ids = [j for j, a in feedback.items() if a == "save"]
    dismissed_ids = [j for j, a in feedback.items() if a == "dismiss"]
    qv = base
    if req.use_feedback and feedback:
        vecs = await _job_vectors(pool, spec.key, spec.dim, saved_ids + dismissed_ids)
        qv = rocchio(
            base,
            [vecs[j] for j in saved_ids if j in vecs],
            [vecs[j] for j in dismissed_ids if j in vecs],
            s.rocchio_alpha, s.rocchio_beta, s.rocchio_gamma,
        )  # fmt: skip

    filters = req.filters.model_copy()
    filters.exclude_job_ids = sorted(set(filters.exclude_job_ids or []) | set(feedback))
    if filters.remote is None and stored.profile.remote_preference == "remote_only":
        filters.remote = True

    t = time.perf_counter()
    async with pool.connection() as conn:
        cands = await vector_search(conn, spec, qv, filters, max(s.rec_candidates, req.k))
    ms["vector"] = round((time.perf_counter() - t) * 1000, 2)

    rows = await hydrate(pool, [c.job_id for c in cands], doc_chars=400)
    scored = []
    for c in cands:
        r = rows.get(c.job_id)
        if not r:
            continue
        m = match_signals(
            c.vector_score, r["skills"], stored.canonical_skills, r["seniority"], stored.profile.seniority,
            s.rec_skill_weight, s.rec_seniority_penalty,
        )  # fmt: skip
        scored.append((m, r))
    scored.sort(key=lambda x: x[0].score, reverse=True)
    scored = cap_per_company(scored, lambda x: x[1]["company"], s.rec_max_per_company)
    ms["total"] = round((time.perf_counter() - t0) * 1000, 2)

    results = [
        Recommendation(
            job_id=r["id"],
            title=r["title"],
            company=r["company"],
            location=r["location"],
            is_remote=r["is_remote"],
            seniority=r["seniority"],
            url=r["url"],
            posted_at=r["posted_at"],
            snippet=" ".join(r["doc"].split())[:280],
            score=round(m.score, 4),
            similarity=round(m.similarity, 4),
            skill_coverage=None if m.skill_coverage is None else round(m.skill_coverage, 3),
            matched_skills=m.matched_skills,
            missing_skills=m.missing_skills,
            seniority_gap=m.seniority_gap,
        )  # fmt: skip
        for m, r in scored[: req.k]
    ]
    return RecommendResponse(
        user_id=req.user_id,
        profile_version=stored.version,
        feedback_used={"save": len(saved_ids), "dismiss": len(dismissed_ids)} if req.use_feedback else {},
        query_shift=0.0 if qv is base else round(1.0 - float(np.dot(base, qv)), 4),
        results=results,
        timings_ms=ms,
    )
