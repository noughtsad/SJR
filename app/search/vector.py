"""Stage 2: approximate nearest-neighbour search with pgvector HNSW.

- `<=>` is cosine distance; similarity = 1 - distance.
- The query has to repeat the partial index's expression and predicate
  (`embedding::vector(dim)` and `model = '<key>'` as literals), or the
  planner can't use the per-model HNSW index.
- Filters + HNSW: the index returns the ef_search nearest rows *before*
  the WHERE clause, so a selective filter can leave too few results.
  pgvector >= 0.8 `hnsw.iterative_scan = relaxed_order` keeps scanning until
  LIMIT is met. Relaxed order can be slightly out of order, so we re-sort in
  an outer query (the pattern from the pgvector docs).
"""

import numpy as np
from pgvector import Vector
from psycopg import AsyncConnection, sql

from app.config import get_settings
from app.embeddings.registry import ModelSpec
from app.search.filters import build_filters
from app.search.models import Candidate, SearchFilters

_SQL = """
WITH nn AS MATERIALIZED (
    SELECT j.id, e.embedding::vector({dim}) <=> %(qv)s::vector({dim}) AS distance
    FROM job_embeddings e
    JOIN jobs j ON j.id = e.job_id
    WHERE e.model = {model} AND {filters}
    ORDER BY e.embedding::vector({dim}) <=> %(qv)s::vector({dim})
    LIMIT %(limit)s
)
SELECT id, 1 - distance AS score FROM nn ORDER BY distance
"""


async def vector_search(
    conn: AsyncConnection, spec: ModelSpec, query_vec: np.ndarray, filters: SearchFilters, limit: int
) -> list[Candidate]:
    where, params = build_filters(filters)
    q = sql.SQL(_SQL).format(dim=sql.Literal(spec.dim), model=sql.Literal(spec.key), filters=where)
    async with conn.transaction():
        # SET LOCAL needs a transaction; values can't be bound parameters.
        ef = int(get_settings().hnsw_ef_search)
        await conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(ef)))
        await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur = await conn.execute(q, {**params, "qv": Vector(query_vec), "limit": limit})
        rows = await cur.fetchall()
    return [Candidate(job_id=r["id"], vector_rank=i, vector_score=float(r["score"])) for i, r in enumerate(rows, 1)]
