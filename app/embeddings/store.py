"""Keeps job_embeddings in sync with jobs, one model at a time.

A job needs (re-)embedding for model M when there is no row for (job, M) or
the stored content_hash differs from jobs.content_hash. Unchanged jobs are
never re-embedded, so a daily run costs as much as that day's changes."""

import logging

from pgvector import Vector
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.embeddings.registry import ModelSpec, get_embedder, get_spec
from app.ingestion.models import JobPosting
from app.ingestion.normalize import embedding_text

log = logging.getLogger(__name__)


def index_name(spec: ModelSpec) -> str:
    return f"job_embeddings_hnsw_{spec.key.replace('-', '_')}"


async def ensure_model_index(pool: AsyncConnectionPool, spec: ModelSpec) -> None:
    """Register the model and build its partial HNSW index over vector(dim)."""
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO embedding_models (key, provider, model_name, dim) VALUES (%s, %s, %s, %s)
               ON CONFLICT (key) DO UPDATE SET provider = EXCLUDED.provider, model_name = EXCLUDED.model_name
               WHERE embedding_models.dim = EXCLUDED.dim""",
            (spec.key, spec.provider, spec.name, spec.dim),
        )
        cur = await conn.execute("SELECT dim FROM embedding_models WHERE key = %s", (spec.key,))
        stored = (await cur.fetchone())["dim"]
        if stored != spec.dim:
            raise RuntimeError(f"model {spec.key} is registered with dim={stored}, spec says {spec.dim}")
        await conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON job_embeddings "
                "USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64) WHERE model = {model}"
            ).format(idx=sql.Identifier(index_name(spec)), dim=sql.Literal(spec.dim), model=sql.Literal(spec.key))
        )


_PENDING = """
SELECT j.id, j.source, j.external_id, j.title, j.company, j.location, j.is_remote,
       j.seniority, j.skills, j.description, j.content_hash
FROM jobs j
LEFT JOIN job_embeddings e ON e.job_id = j.id AND e.model = %(model)s
WHERE j.is_active AND (e.job_id IS NULL OR e.content_hash <> j.content_hash)
ORDER BY j.id
LIMIT %(limit)s
"""


async def embed_pending(pool: AsyncConnectionPool, model_key: str | None = None, limit: int | None = None) -> int:
    spec = get_spec(model_key)
    embedder = get_embedder(spec.key)
    await ensure_model_index(pool, spec)
    batch_size = get_settings().embedding_batch_size
    done = 0
    while limit is None or done < limit:
        take = batch_size if limit is None else min(batch_size, limit - done)
        async with pool.connection() as conn:
            cur = await conn.execute(_PENDING, {"model": spec.key, "limit": take})
            rows = await cur.fetchall()
        if not rows:
            break
        texts = [
            embedding_text(
                JobPosting.model_construct(
                    **{k: r[k] for k in ("source", "external_id", "title", "company", "location",
                                         "is_remote", "seniority", "skills", "description")}  # fmt: skip
                )
            )
            for r in rows
        ]
        vectors = await embedder.embed_documents(texts)
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO job_embeddings (job_id, model, content_hash, embedding)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (job_id, model) DO UPDATE
                   SET content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding, created_at = now()""",
                [(r["id"], spec.key, r["content_hash"], Vector(v)) for r, v in zip(rows, vectors, strict=True)],
            )
        done += len(rows)
        log.info("embedded %d jobs with %s (total %d)", len(rows), spec.key, done)
    return done
