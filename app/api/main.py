import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.agent.memory import delete_fact, get_facts
from app.agent.runtime import agent_runtime, chat_stream, thread_config
from app.config import get_settings
from app.db import make_pool
from app.embeddings.registry import get_embedder
from app.llm import LLMOutputError
from app.profile.extract import extract_profile, read_resume
from app.profile.models import StoredProfile
from app.profile.store import FeedbackAction, NotFound, get_profile, save_profile, set_feedback
from app.recommend.explain import Explanation, explain
from app.recommend.service import RecommendRequest, RecommendResponse, recommend
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
    async with agent_runtime(pool) as (graph, store):
        app.state.agent, app.state.store = graph, store
        yield
    await pool.close()


app = FastAPI(title="Semantic Job Search", version="0.3.0", lifespan=lifespan)

MAX_RESUME_BYTES = 5 * 1024 * 1024


@app.exception_handler(NotFound)
async def _not_found(_request: Request, e: NotFound):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(LLMOutputError)
async def _llm_bad_output(_request: Request, e: LLMOutputError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=502, content={"detail": f"LLM returned invalid output after retries: {e}"})


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


# ---------------------------------------------------------------------------
# Layer 3: profile, recommendations, feedback, explanations
# ---------------------------------------------------------------------------


@app.post("/users/{user_id}/resume", response_model=StoredProfile)
async def upload_resume(user_id: str, file: UploadFile, request: Request):
    data = await file.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "resume larger than 5 MB")
    try:
        text = read_resume(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    profile = await extract_profile(text)
    pool = request.app.state.pool
    await save_profile(pool, user_id, profile, text)
    return await get_profile(pool, user_id)


@app.get("/users/{user_id}/profile", response_model=StoredProfile)
async def read_profile(user_id: str, request: Request):
    return await get_profile(request.app.state.pool, user_id)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_post(req: RecommendRequest, request: Request):
    return await recommend(request.app.state.pool, req)


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    job_id: int
    action: FeedbackAction | None = Field(description="save | dismiss | null to clear")


@app.post("/feedback", status_code=204)
async def feedback(req: FeedbackRequest, request: Request):
    await set_feedback(request.app.state.pool, req.user_id, req.job_id, req.action)


@app.get("/users/{user_id}/jobs/{job_id}/explain", response_model=Explanation)
async def explain_match(user_id: str, job_id: int, request: Request, refresh: bool = False):
    return await explain(request.app.state.pool, user_id, job_id, refresh=refresh)


# ---------------------------------------------------------------------------
# Layer 4: career agent
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    thread_id: str = Field("default", min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=20_000)  # long enough for a pasted resume


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """Server-sent events: token, tool_call, tool_result, retract, then final (or error).
    See app.agent.runtime for the event shapes."""
    import json

    from fastapi.responses import StreamingResponse

    async def events():
        async for ev in chat_stream(request.app.state.agent, req.user_id, req.thread_id, req.message):
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/chat/{user_id}/{thread_id}")
async def chat_history(user_id: str, thread_id: str, request: Request):
    """The visible conversation: user messages and final answers (tool traffic omitted)."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.graph import _text

    state = await request.app.state.agent.aget_state(thread_config(user_id, thread_id))
    out = []
    for m in (state.values or {}).get("messages", []):
        if isinstance(m, HumanMessage) and not m.additional_kwargs.get("validator"):
            out.append({"role": "user", "text": _text(m.content)})
        elif isinstance(m, AIMessage) and not m.tool_calls and _text(m.content):
            out.append({"role": "assistant", "text": _text(m.content)})
    return {"thread_id": thread_id, "messages": out}


@app.get("/users/{user_id}/facts")
async def list_facts(user_id: str, request: Request):
    return await get_facts(request.app.state.store, user_id)


@app.delete("/users/{user_id}/facts/{key}", status_code=204)
async def forget_fact(user_id: str, key: str, request: Request):
    await delete_fact(request.app.state.store, user_id, key)
