"""Thin LLM wrapper (Gemini). Every call is schema-validated against a Pydantic
model and retried on API errors *and* on output that fails validation."""

from functools import cache

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

DEFAULT_MODEL = "gemini-2.5-flash"


@cache
def _client():
    from google import genai

    key = get_settings().gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    # v1beta: system_instruction and response_schema aren't available on v1
    return genai.Client(api_key=key, http_options={"api_version": "v1beta"})


def _retryable(e: BaseException) -> bool:
    """Retry rate limits, server errors and invalid output; not other 4xx (bad key, bad request)."""
    from google.genai import errors

    if isinstance(e, LLMOutputError | errors.ServerError):
        return True
    return isinstance(e, errors.ClientError) and e.code == 429


@cache
def rate_limiter():
    """One process-wide limiter shared by every Gemini call (agent, extraction, explanations,
    eval judge), so together they stay under a requests-per-minute quota (LLM_RPM).
    Waiting is cheaper than tripping 429s and backing off."""
    rpm = get_settings().llm_rpm
    if not rpm:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.05, max_bucket_size=1)


class LLMOutputError(Exception):
    pass


async def generate_structured[T: BaseModel](
    prompt: str, schema: type[T], *, model: str = DEFAULT_MODEL, temperature: float = 0.7, system: str | None = None
) -> T:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        system_instruction=system,
    )
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_exception(_retryable),
        reraise=True,
    ):
        with attempt:
            if limiter := rate_limiter():
                await limiter.aacquire()
            resp = await _client().aio.models.generate_content(model=model, contents=prompt, config=config)
            try:
                return schema.model_validate_json(resp.text or "")
            except ValidationError as e:
                raise LLMOutputError(f"output failed {schema.__name__} validation: {e}") from e
    raise AssertionError("unreachable")
