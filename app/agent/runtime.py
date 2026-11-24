"""Agent lifecycle (checkpointer + store on Postgres) and the streaming chat loop.

Stream events (also the /chat SSE payloads):
  {"type": "token", "text": ...}                     answer text as it is generated
  {"type": "tool_call", "name": ..., "args": {...}}
  {"type": "tool_result", "name": ..., "preview": ...}
  {"type": "retract", "reason": ...}                 the streamed draft failed the citation check;
                                                     clients discard it, a corrected answer follows
  {"type": "final", "text": ..., "cited_job_ids": [...], "usage": {...}, "latency_ms": ...}
  {"type": "error", "kind": "rate_limited" | "failed", "message": ...}   instead of final
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agent.graph import _text, build_graph, cited_job_ids
from app.config import get_settings

log = logging.getLogger(__name__)


def _error_kind(e: BaseException) -> str:
    text = f"{type(e).__name__} {e}"
    return "rate_limited" if "RateLimit" in text or "RESOURCE_EXHAUSTED" in text or " 429" in text else "failed"


@asynccontextmanager
async def agent_runtime(app_pool: AsyncConnectionPool):
    """Yields (graph, store). LangGraph's Postgres savers need autocommit and no
    prepared statements, so they get their own small pool."""
    s = get_settings()
    mem_pool = AsyncConnectionPool(
        s.database_url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await mem_pool.open()
    try:
        saver, store = AsyncPostgresSaver(mem_pool), AsyncPostgresStore(mem_pool)
        await saver.setup()  # idempotent migrations
        await store.setup()
        yield build_graph(app_pool, saver, store), store
    finally:
        await mem_pool.close()


def thread_config(user_id: str, thread_id: str) -> dict:
    # Thread ids are namespaced by user, so guessing another user's thread id reads nothing.
    return {
        "configurable": {"user_id": user_id, "thread_id": f"{user_id}:{thread_id}"},
        "recursion_limit": get_settings().agent_recursion_limit,
    }


def _usage(messages) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    for m in messages:
        u = getattr(m, "usage_metadata", None)
        if isinstance(m, AIMessage) and u:
            total["input_tokens"] += u.get("input_tokens", 0)
            total["output_tokens"] += u.get("output_tokens", 0)
            total["llm_calls"] += 1
    return total


async def chat_stream(graph, user_id: str, thread_id: str, text: str) -> AsyncIterator[dict]:
    config = thread_config(user_id, thread_id)
    t0 = time.perf_counter()
    turn_msgs = []  # every AI/tool message produced in this turn, for usage accounting
    inp = {"messages": [HumanMessage(text)], "citation_retries": 0}
    try:
        async for ev in _run(graph, inp, config, turn_msgs):
            yield ev
    except Exception as e:  # the LLM call itself failed after its own retries
        log.exception("agent turn failed")
        kind = _error_kind(e)
        msg = "The language model is rate-limited; try again later." if kind == "rate_limited" else str(e)[:300]
        yield {"type": "error", "kind": kind, "message": msg}
        return

    state = await graph.aget_state(config)
    final = next((m for m in reversed(state.values["messages"]) if isinstance(m, AIMessage)), None)
    answer = _text(final.content) if final else ""
    yield {
        "type": "final",
        "text": answer,
        "cited_job_ids": cited_job_ids(answer),
        "usage": _usage(turn_msgs),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }


async def _run(graph, inp, config, turn_msgs) -> AsyncIterator[dict]:
    async for mode, data in graph.astream(inp, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk):
                if t := _text(chunk.content):
                    yield {"type": "token", "text": t}
            continue
        for update in (data or {}).values():
            for m in (update or {}).get("messages", []):
                if isinstance(m, AIMessage):
                    turn_msgs.append(m)
                    for tc in m.tool_calls:
                        yield {"type": "tool_call", "name": tc["name"], "args": tc["args"]}
                elif isinstance(m, ToolMessage):
                    yield {"type": "tool_result", "name": m.name, "preview": _text(m.content)[:300]}
                elif isinstance(m, HumanMessage) and m.additional_kwargs.get("validator"):
                    yield {"type": "retract", "reason": _text(m.content)}
