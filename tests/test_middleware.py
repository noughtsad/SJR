import asyncio
import json
import logging

from app.api.middleware import JsonFormatter, RateLimiter, RequestContextMiddleware, request_id_var, route_class


def test_token_bucket_allows_burst_then_refills():
    rl = RateLimiter({"default": 60})  # 1 token per second, burst of 60
    assert all(rl.take("a", "default", now=0.0) == 0 for _ in range(60))
    wait = rl.take("a", "default", now=0.0)
    assert 0.9 < wait <= 1.0
    assert rl.take("a", "default", now=1.01) == 0  # refilled one token
    assert rl.take("b", "default", now=0.0) == 0  # other clients unaffected


def test_unlimited_class_and_route_classes():
    rl = RateLimiter({"default": 1})
    assert all(rl.take("a", "free", now=0.0) == 0 for _ in range(100))
    assert route_class("POST", "/chat") == "llm"
    assert route_class("GET", "/chat/u/t") == "default"
    assert route_class("POST", "/users/u/resume") == "llm"
    assert route_class("GET", "/users/u/jobs/1/explain") == "llm"
    assert route_class("GET", "/health") == "free"
    assert route_class("POST", "/search") == "default"


def test_json_formatter_includes_request_id_and_extras():
    token = request_id_var.set("abc123")
    try:
        rec = logging.LogRecord("app.x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        rec.duration_ms = 12.5
        out = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert out["msg"] == "hello world" and out["request_id"] == "abc123" and out["duration_ms"] == 12.5


def _call(mw, path="/search", method="POST", headers=()):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "method": method, "path": path, "headers": list(headers), "client": ("1.2.3.4", 1)}
    asyncio.run(mw(scope, receive, send))
    return sent


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_middleware_sets_request_id_and_limits():
    mw = RequestContextMiddleware(_ok_app, RateLimiter({"llm": 2}))
    first = _call(mw, "/chat", headers=[(b"x-request-id", b"req-1")])
    assert first[0]["status"] == 200 and (b"x-request-id", b"req-1") in first[0]["headers"]
    _call(mw, "/chat")
    third = _call(mw, "/chat")
    assert third[0]["status"] == 429
    assert any(k == b"retry-after" for k, _ in third[0]["headers"])
    assert _call(mw, "/search")[0]["status"] == 200  # different route class, own budget


def test_forwarded_for_only_when_trusted():
    hdr = [(b"x-forwarded-for", b"9.9.9.9, 10.0.0.1")]
    assert (
        RequestContextMiddleware(_ok_app, trust_proxy=True)._client({"headers": hdr, "client": ("1.1.1.1", 0)})
        == "9.9.9.9"
    )
    assert RequestContextMiddleware(_ok_app)._client({"headers": hdr, "client": ("1.1.1.1", 0)}) == "1.1.1.1"
