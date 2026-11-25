"""Request plumbing: request ids, structured logs, access log, rate limiting.

Written as plain ASGI middleware (not BaseHTTPMiddleware) so streamed /chat
responses pass through untouched instead of being buffered.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
log = logging.getLogger("app.access")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """One JSON object per line; `extra={...}` fields are merged in."""

    _STD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in self._STD})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(fmt: str = "text", level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
        handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):  # our access log replaces uvicorn's
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Rate limiting: token bucket per (client, route class), in process memory.
# Fine for one instance; several replicas would need a shared store (Redis).
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, limits_per_min: dict[str, int], max_keys: int = 50_000):
        self.limits = limits_per_min
        self.buckets: dict[tuple[str, str], Bucket] = {}
        self.max_keys = max_keys

    def take(self, client: str, route_class: str, now: float | None = None) -> float:
        """Consume one token. Returns 0 if allowed, else seconds until a token is available."""
        limit = self.limits.get(route_class)
        if not limit:
            return 0.0
        now = time.monotonic() if now is None else now
        rate = limit / 60.0
        key = (client, route_class)
        b = self.buckets.get(key)
        if b is None:
            if len(self.buckets) >= self.max_keys:  # crude memory bound: forget everyone
                self.buckets.clear()
            b = self.buckets[key] = Bucket(tokens=float(limit), updated=now)
        b.tokens = min(float(limit), b.tokens + (now - b.updated) * rate)
        b.updated = now
        if b.tokens >= 1:
            b.tokens -= 1
            return 0.0
        return (1 - b.tokens) / rate


def route_class(method: str, path: str) -> str:
    """LLM-backed endpoints cost money and quota, so they get a much tighter limit."""
    if path == "/chat" and method == "POST":
        return "llm"
    if path.endswith("/resume") or path.endswith("/explain"):
        return "llm"
    if path in ("/health", "/docs", "/openapi.json"):
        return "free"
    return "default"


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


class RequestContextMiddleware:
    def __init__(self, app, limiter: RateLimiter | None = None, trust_proxy: bool = False):
        self.app, self.limiter, self.trust_proxy = app, limiter, trust_proxy

    def _client(self, scope) -> str:
        if self.trust_proxy:
            for k, v in scope.get("headers", []):
                if k == b"x-forwarded-for":
                    return v.decode().split(",")[0].strip()
        return (scope.get("client") or ("unknown", 0))[0]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        rid = headers.get(b"x-request-id", b"").decode()[:64] or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        method, path = scope["method"], scope["path"]
        client = self._client(scope)
        t0 = time.perf_counter()
        status = 500

        async def send_wrapped(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", rid.encode())]
            await send(message)

        try:
            rc = route_class(method, path)
            wait = self.limiter.take(client, rc) if self.limiter else 0.0
            if wait > 0:
                status = 429
                body = json.dumps(
                    {"detail": f"rate limit exceeded for {rc} endpoints", "retry_after_s": round(wait, 1)}
                )
                await send({
                    "type": "http.response.start", "status": 429,
                    "headers": [(b"content-type", b"application/json"), (b"retry-after", str(int(wait) + 1).encode()),
                                (b"x-request-id", rid.encode())],
                })  # fmt: skip
                await send({"type": "http.response.body", "body": body.encode()})
                return
            await self.app(scope, receive, send_wrapped)
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            log.info(
                "%s %s %s %.0fms", method, path, status, ms,
                extra={"method": method, "path": path, "status": status, "duration_ms": ms, "client": client},
            )  # fmt: skip
            request_id_var.reset(token)
