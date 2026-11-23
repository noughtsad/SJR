import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from app.config import get_settings
from app.db import make_pool
from app.embeddings.registry import get_embedder
from app.search.models import SearchFilters, SearchMode, SearchRequest, SearchResponse
from app.search.rerank import warmup as rerank_warmup
from app.search.service import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = make_pool()
    await pool.open()
    app.state.pool = pool
    # load models at startup, not on the first request
    get_embedder(get_settings().embedding_model)
    rerank_warmup()
    yield
    await pool.close()


app = FastAPI(title="Semantic Job Search", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    async with request.app.state.pool.connection() as conn:
        cur = await conn.execute(
            """SELECT (SELECT count(*) FROM jobs WHERE is_active) AS active_jobs,
                      (SELECT max(finished_at) FROM ingestion_runs WHERE status = 'ok') AS last_ingest"""
        )
        row = await cur.fetchone()
    return {"status": "ok", **row}


@app.post("/search", response_model=SearchResponse)
async def search_post(req: SearchRequest, request: Request):
    try:
        return await search(request.app.state.pool, req)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/search", response_model=SearchResponse)
async def search_get(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    mode: SearchMode = SearchMode.hybrid_rerank,
    k: int = Query(10, ge=1, le=50),
    model: str | None = None,
    remote: bool | None = None,
    location: str | None = None,
    seniority: list[str] | None = Query(None),
    skills: list[str] | None = Query(None),
    posted_within_days: int | None = Query(None, ge=1),
):
    filters = SearchFilters(
        remote=remote, location=location, seniority=seniority, skills=skills, posted_within_days=posted_within_days
    )
    return await search_post(SearchRequest(query=q, mode=mode, k=k, model=model, filters=filters), request)
