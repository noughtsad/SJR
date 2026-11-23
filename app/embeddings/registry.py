"""Swappable embedding models. Add an entry here to compare a new model;
`embed --model <key>` backfills it and search takes `model=<key>`."""

import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np

from app.config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str  # "local" (sentence-transformers) | "gemini"
    name: str
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        # BGE v1.5 recommends an instruction prefix on short queries only.
        ModelSpec("bge-small", "local", "BAAI/bge-small-en-v1.5", 384,
                  query_prefix="Represent this sentence for searching relevant passages: "),
        ModelSpec("minilm", "local", "sentence-transformers/all-MiniLM-L6-v2", 384),
        ModelSpec("gemini", "gemini", "gemini-embedding-001", 768),
    ]
}  # fmt: skip

_KEY_OK = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
for _k in MODELS:
    assert _KEY_OK.match(_k), f"model key {_k!r} is used in SQL identifiers; keep it [a-z0-9_-]"


class Embedder(Protocol):
    spec: ModelSpec

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...
    async def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    def __init__(self, spec: ModelSpec):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.name, device="cpu")

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        out = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return list(out.astype(np.float32))

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return await asyncio.to_thread(self._encode, [self.spec.doc_prefix + t for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return (await asyncio.to_thread(self._encode, [self.spec.query_prefix + text]))[0]


class GeminiEmbedder:
    MAX_BATCH = 100

    def __init__(self, spec: ModelSpec):
        from google import genai

        key = get_settings().gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.spec = spec
        self.client = genai.Client(api_key=key)

    async def _embed(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        from google.genai import types
        from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            chunk = texts[i : i + self.MAX_BATCH]
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True
            ):
                with attempt:
                    resp = await self.client.aio.models.embed_content(
                        model=self.spec.name,
                        contents=chunk,
                        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.spec.dim),
                    )
            for e in resp.embeddings:
                v = np.asarray(e.values, dtype=np.float32)
                out.append(v / (np.linalg.norm(v) or 1.0))  # truncated dims are not unit length
        return out

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


class CachedQueryEmbedder:
    """Small LRU in front of embed_query: repeated searches skip the model call."""

    def __init__(self, inner: Embedder, size: int = 1024):
        self.inner, self.spec, self.size = inner, inner.spec, size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return await self.inner.embed_documents(texts)

    async def embed_query(self, text: str) -> np.ndarray:
        key = " ".join(text.lower().split())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        v = await self.inner.embed_query(text)
        self._cache[key] = v
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return v


def get_spec(key: str | None = None) -> ModelSpec:
    key = key or get_settings().embedding_model
    if key not in MODELS:
        raise ValueError(f"unknown embedding model {key!r}; choose from {sorted(MODELS)}")
    return MODELS[key]


@cache
def get_embedder(key: str | None = None) -> Embedder:
    spec = get_spec(key)
    inner = LocalEmbedder(spec) if spec.provider == "local" else GeminiEmbedder(spec)
    return CachedQueryEmbedder(inner)
