"""Optional Langfuse tracing. Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set,
and every function here is a cheap no-op when it's off, so nothing else depends on Langfuse.

What gets traced:
- Agent turns: a LangChain callback on each graph run gives one trace per turn, with nested
  spans for every node, tool call and Gemini call (tokens, latency), tagged with
  user id (Langfuse "user") and thread id (Langfuse "session").
- Direct Gemini calls (profile extraction, match explanations, the eval judge) go through
  app.llm.generate_structured, which records a "generation" per call.
"""

from contextlib import contextmanager
from functools import cache

from app.config import get_settings


@cache
def client():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
        environment=s.langfuse_environment,
    )


def enabled() -> bool:
    return client() is not None


def agent_callbacks(user_id: str, thread_id: str) -> tuple[list, dict]:
    """(callbacks, metadata) to merge into a LangGraph run config."""
    if not enabled():
        return [], {}
    from langfuse.langchain import CallbackHandler

    meta = {"langfuse_user_id": user_id, "langfuse_session_id": f"{user_id}:{thread_id}", "langfuse_tags": ["chat"]}
    return [CallbackHandler(public_key=get_settings().langfuse_public_key)], meta


class _NoOp:
    def update(self, **_kwargs):
        pass


@contextmanager
def generation(name: str, model: str, prompt: str, system: str | None):
    """Wrap one direct LLM call. Yields an object with .update(output=..., usage_details=...)."""
    lf = client()
    if lf is None:
        yield _NoOp()
        return
    with lf.start_as_current_observation(
        name=name, as_type="generation", model=model, input={"system": system, "prompt": prompt[:20_000]}
    ) as gen:
        yield gen


def flush() -> None:
    if (lf := client()) is not None:
        lf.flush()


async def aflush() -> None:
    """Send buffered spans before the response completes. Needed on Cloud Run: once a request
    ends the instance's CPU is throttled, so the SDK's background exporter would never run and
    traces would sit in memory until the next request (or be lost at scale-down)."""
    if client() is not None:
        import asyncio

        await asyncio.to_thread(flush)
