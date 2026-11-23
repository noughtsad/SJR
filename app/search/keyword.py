"""Stage 1: keyword search on Postgres full-text search, with IDF.

Postgres' ts_rank/ts_rank_cd have no notion of term rarity. A query like
"I want a physician assistant role in general surgery" OR-ed lexeme by lexeme
matched 15.6k of 19.4k jobs (because of "want", "role", "general") and spent
~700 ms ranking them, while an AND query matched nothing.

So, BM25-lite:
1. lexeme_stats holds each lexeme's document frequency (refreshed after ingest).
2. Query lexemes found in more than `keyword_max_df` of documents are dropped.
   They're near-stopwords for this corpus, and they are what blew up the
   candidate set. If every term is that common, the rarest two are kept.
3. Candidates match ANY remaining term (GIN index).
4. score = sum over matched terms of idf(t) * (2 if t is in the title else 1),
   with idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), BM25's IDF.

Not modelled: term-frequency saturation and length normalisation (full BM25).
Lexemes are cast straight to tsquery (`'x'::tsquery`) rather than via
to_tsquery('english', ...), because re-stemming an already-stemmed lexeme can
change it.
"""

from psycopg import AsyncConnection, sql

from app.config import get_settings
from app.search.filters import build_filters
from app.search.models import Candidate, SearchFilters

_SQL = """
WITH n AS (SELECT n_docs FROM corpus_stats),
q_terms AS (
    SELECT DISTINCT l.lexeme, s.ndoc,
           ln(1 + (n.n_docs - s.ndoc + 0.5) / (s.ndoc + 0.5)) AS idf
    FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS l(lexeme)
    JOIN lexeme_stats s ON s.lexeme = l.lexeme, n
),
terms AS (
    SELECT lexeme, idf FROM q_terms WHERE ndoc <= %(max_df)s * (SELECT n_docs FROM n)
    UNION
    -- every term is common: fall back to the two rarest rather than return nothing
    SELECT lexeme, idf FROM (SELECT * FROM q_terms ORDER BY ndoc LIMIT 2) r
    WHERE NOT EXISTS (SELECT 1 FROM q_terms WHERE ndoc <= %(max_df)s * (SELECT n_docs FROM n))
),
tq AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS q FROM terms
),
cand AS (
    SELECT j.id, j.search_tsv, j.posted_at
    FROM jobs j, tq
    WHERE tq.q IS NOT NULL AND j.search_tsv @@ tq.q AND {filters}
)
SELECT c.id,
       (SELECT sum(t.idf * CASE WHEN ts_filter(c.search_tsv, '{{a}}') @@ quote_literal(t.lexeme)::tsquery
                                THEN 2 ELSE 1 END)
        FROM terms t
        WHERE c.search_tsv @@ quote_literal(t.lexeme)::tsquery) AS score
FROM cand c
ORDER BY score DESC, c.posted_at DESC NULLS LAST
LIMIT %(limit)s
"""


async def keyword_search(conn: AsyncConnection, query: str, filters: SearchFilters, limit: int) -> list[Candidate]:
    where, params = build_filters(filters)
    cur = await conn.execute(
        sql.SQL(_SQL).format(filters=where),
        {**params, "query": query, "limit": limit, "max_df": get_settings().keyword_max_df},
    )
    return [
        Candidate(job_id=r["id"], keyword_rank=i, keyword_score=float(r["score"]))
        for i, r in enumerate(await cur.fetchall(), start=1)
    ]


async def refresh_lexeme_stats(conn: AsyncConnection) -> int:
    """Recompute document frequencies over active jobs (~1.6 s at 19k jobs)."""
    async with conn.transaction():
        await conn.execute("TRUNCATE lexeme_stats")
        cur = await conn.execute(
            """INSERT INTO lexeme_stats (lexeme, ndoc)
               SELECT word, ndoc FROM ts_stat('SELECT search_tsv FROM jobs WHERE is_active')"""
        )
        n_lexemes = cur.rowcount
        await conn.execute(
            """INSERT INTO corpus_stats (id, n_docs, refreshed_at)
               VALUES (TRUE, (SELECT count(*) FROM jobs WHERE is_active), now())
               ON CONFLICT (id) DO UPDATE SET n_docs = EXCLUDED.n_docs, refreshed_at = now()"""
        )
    return n_lexemes
