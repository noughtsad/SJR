"""Stage 1: Postgres full-text search.

websearch_to_tsquery ANDs every term, which returns nothing for a natural
query like "python backend engineer fintech". So the query is turned into
lexemes with the same 'english' config used to build search_tsv, and the
lexemes are OR-ed. ts_rank_cd then rewards documents that match more terms,
closer together, in higher-weighted fields (title=A > company/category=B >
description=C). Normalisation flag 32 maps the rank into 0..1 (rank/(rank+1)).
"""

from psycopg import AsyncConnection, sql

from app.search.filters import build_filters
from app.search.models import Candidate, SearchFilters

_SQL = """
WITH q AS (
    SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS tsq
    FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
)
SELECT j.id, ts_rank_cd('{{0.1, 0.2, 0.4, 1.0}}', j.search_tsv, q.tsq, 32) AS score
FROM jobs j, q
WHERE q.tsq IS NOT NULL AND j.search_tsv @@ q.tsq AND {filters}
ORDER BY score DESC, j.posted_at DESC NULLS LAST
LIMIT %(limit)s
"""


async def keyword_search(conn: AsyncConnection, query: str, filters: SearchFilters, limit: int) -> list[Candidate]:
    where, params = build_filters(filters)
    cur = await conn.execute(sql.SQL(_SQL).format(filters=where), {**params, "query": query, "limit": limit})
    return [
        Candidate(job_id=r["id"], keyword_rank=i, keyword_score=float(r["score"]))
        for i, r in enumerate(await cur.fetchall(), start=1)
    ]
