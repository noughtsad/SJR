"""Orchestrates the stages and records per-stage latency."""

import asyncio
import time
from contextlib import contextmanager

from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.embeddings.registry import get_embedder, get_spec
from app.search.fusion import rrf_fuse
from app.search.keyword import keyword_search
from app.search.models import Candidate, JobHit, SearchMode, SearchRequest, SearchResponse
from app.search.rerank import rerank
from app.search.rerank import warmup as rerank_warmup
from app.search.vector import vector_search


class _Timer:
    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        yield
        self.ms[name] = round((time.perf_counter() - t0) * 1000, 2)


_HYDRATE = """
SELECT id, title, company, location, is_remote, seniority, skills, url, posted_at,
       left(description, %(doc_chars)s) AS doc
FROM jobs WHERE id = ANY(%(ids)s)
"""


async def hydrate(pool: AsyncConnectionPool, ids: list[int], doc_chars: int = 2000) -> dict[int, dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(_HYDRATE, {"ids": ids, "doc_chars": doc_chars})
        return {r["id"]: r for r in await cur.fetchall()}


def _rerank_text(r: dict) -> str:
    head = f"{r['title']} at {r['company'] or 'unknown'} ({r['location'] or ''}{', remote' if r['is_remote'] else ''})"
    return f"{head}\n{r['doc']}"


async def search(pool: AsyncConnectionPool, req: SearchRequest) -> SearchResponse:
    s = get_settings()
    t = _Timer()
    mode = req.mode
    uses_vector = mode is not SearchMode.keyword
    spec = get_spec(req.model) if uses_vector else None
    # Resolve (and on first use, load) the model before timing starts: loading is
    # synchronous and would block the event loop and inflate every stage's timing.
    embedder = await asyncio.to_thread(get_embedder, spec.key) if spec else None
    if mode is SearchMode.hybrid_rerank:
        await asyncio.to_thread(rerank_warmup)  # no-op once loaded
    pool_size = max(s.candidate_pool, req.k)

    async def run_keyword() -> list[Candidate]:
        with t("keyword"):
            async with pool.connection() as conn:
                return await keyword_search(conn, req.query, req.filters, pool_size)

    async def run_vector() -> list[Candidate]:
        with t("embed_query"):
            qv = await embedder.embed_query(req.query)
        with t("vector"):
            async with pool.connection() as conn:
                return await vector_search(conn, spec, qv, req.filters, pool_size)

    with t("total"):
        if mode is SearchMode.keyword:
            ranked = await run_keyword()
        elif mode is SearchMode.vector:
            ranked = await run_vector()
        else:
            kw, vec = await asyncio.gather(run_keyword(), run_vector())  # two pool connections, in parallel
            with t("fusion"):
                ranked = rrf_fuse(kw, vec, k=s.rrf_k, w_keyword=s.rrf_w_keyword, w_vector=s.rrf_w_vector)

        top_n = s.rerank_top_n if mode is SearchMode.hybrid_rerank else req.k
        ranked = ranked[:top_n]
        with t("hydrate"):
            rows = await hydrate(pool, [c.job_id for c in ranked])
        ranked = [c for c in ranked if c.job_id in rows]  # job could expire between stages

        if mode is SearchMode.hybrid_rerank:
            with t("rerank"):
                ranked = await rerank(req.query, ranked, {i: _rerank_text(r) for i, r in rows.items()})

    results = []
    for c in ranked[: req.k]:
        r = rows[c.job_id]
        results.append(
            JobHit(
                **c.model_dump(),
                title=r["title"],
                company=r["company"],
                location=r["location"],
                is_remote=r["is_remote"],
                seniority=r["seniority"],
                skills=r["skills"],
                url=r["url"],
                posted_at=r["posted_at"],
                snippet=" ".join(r["doc"].split())[:280],
            )  # fmt: skip
        )
    return SearchResponse(
        query=req.query, mode=mode, model=spec.key if spec else None, results=results, timings_ms=t.ms
    )
