# Semantic Job Search

Semantic job search and recommendation over a continuously changing jobs index, with a career agent on top (in progress).

| Layer | Status |
|---|---|
| 1. Ingestion + changing index (Postgres + pgvector) | ✅ |
| 2. Search: keyword → vector → hybrid (RRF) → cross-encoder rerank | ✅ |
| 3. Resume profile, feedback loop, explanations | ⏳ |
| 4. LangGraph career agent (tools, memory) | ⏳ |
| 5. Retrieval + agent evals in CI | ⏳ |
| 6. Production hardening, tracing, Next.js UI | partial (FastAPI, Docker, `/health`, query cache) |

## Quickstart

```bash
docker compose up -d db                       # Postgres 17 + pgvector on localhost:5433
python -m venv .venv && .venv/Scripts/activate  # (source .venv/bin/activate on Linux/macOS)
pip install -r requirements-dev.txt
cp .env.example .env

python -m app.cli init-db
python -m app.cli ingest hn                   # HN "Who is hiring" (last 2 monthly threads, ~450 jobs)
python -m app.cli ingest remotive             # Remotive public API
python -m app.cli ingest kaggle --path data/postings.csv --limit 20000   # optional bulk data
python -m app.cli embed                       # embeds only new/changed jobs
python -m app.cli search "LLM agents and retrieval in python" --mode hybrid_rerank
python -m app.serve                           # API on :8000  (docs at /docs)
```

API: `GET /search?q=...&mode=hybrid_rerank&remote=true&seniority=senior`, `POST /search` (JSON body), `GET /health`.

## Architecture

```
 sources (Remotive, HN Algolia, Kaggle CSV)
     │  fetch_raw → to_posting (Pydantic JobPosting)
     ▼
 pipeline: validate → dedupe → diff by content_hash → insert / update / touch → expire
     ▼
 jobs (metadata + generated tsvector)      job_embeddings (job_id, model, content_hash, vector)
     │                                            │   partial HNSW index per model
     ▼                                            ▼
 keyword (FTS, ts_rank_cd)        vector (cosine, HNSW, iterative scan + filters)
            └──────────── RRF fusion (k=60) ────────────┘
                              ▼
                cross-encoder rerank of top 30
```

## Design decisions

**Changing index**
- Every posting gets a `content_hash` (sha256 of the exact text that gets embedded). Re-ingesting classifies each row as new, changed or unchanged. A changed hash marks the job for re-embedding, and unchanged rows only get `last_seen_at` bumped. A second run of `embed` therefore does nothing.
- **Expiry** depends on the source. Snapshot sources (Remotive, HN) expire anything not seen in a successful run. A guard skips expiry if a run sees fewer than 50% of the currently active jobs, since a partial API response shouldn't delist the index. Incremental sources expire by TTL. The historical Kaggle data never expires. Rows are soft-expired (`is_active = false`), so evals and feedback history keep their foreign keys.
- **Cross-source dedupe**: `dedupe_key` = hash of normalised title|company|location. HN companies often repost across months, which accounts for the 38 duplicates on the first run.
- Every run writes counts to `ingestion_runs` (`python -m app.cli stats`).

**Swappable embeddings**: `job_embeddings` stores vectors from several models side by side in one untyped `vector` column. Each model gets a partial expression index:
`CREATE INDEX ... USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE model = 'bge-small'`.
Queries repeat that expression and predicate as literals so the planner can match the index. Adding a model means one `ModelSpec` line plus `embed --model <key>`. Models available: `bge-small` (default), `minilm`, `gemini` (`gemini-embedding-001`, 768d).
> At ~500 rows Postgres correctly prefers an exact scan (cheaper than HNSW). The HNSW index takes over at Kaggle scale, which you can check with `EXPLAIN`.

**Search stages**: each stage is a separate module, and each response returns every stage's rank and score plus per-stage latency.
- *Keyword*: `websearch_to_tsquery` ANDs all terms, so natural-language queries match nothing. Instead the query is lexemised with the same `english` config and the lexemes are OR-ed. `ts_rank_cd` with field weights (title A > company/category B > body C) then rewards documents that match more terms.
- *Vector*: filters plus HNSW can return too few rows, because the index returns `ef_search` neighbours before the WHERE clause runs. pgvector 0.8's `hnsw.iterative_scan = relaxed_order` keeps scanning until the limit is met, and an outer `ORDER BY` restores exact order.
- *Fusion*: RRF uses only ranks, because `ts_rank` and cosine similarity are on incomparable scales. Score = Σ 1/(60 + rank).
- *Rerank*: a cross-encoder reads query and job together, which is more accurate than a bi-encoder but too slow for the whole index, so it only rescores the fused top 30.

**Observed latency (laptop CPU, 465 jobs)**: keyword ~5 ms, vector ~6 ms, query embedding ~15–50 ms (≈0 on a cache hit), hybrid total ~70 ms, rerank ~1 s. The reranker dominates. Options are a smaller top-N, ONNX/quantised models, or a GPU.

## Data sources and terms
- **Remotive**: the public API currently lists only a handful of jobs and asks for ≤4 calls/day. Raw responses are saved to `data/raw/` and can be replayed with `--from-file`. Their terms require linking back to the Remotive URL and crediting Remotive.
- **HN Who is hiring**: hn.algolia.com, no key needed. Headers are free text, so title/location parsing is heuristic.
- **Kaggle**: [arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) `postings.csv`. Other CSVs work with a column mapping.

## Scheduling
`.github/workflows/refresh-index.yml` runs `python -m app.cli refresh` 3× a day against the `DATABASE_URL` secret. `ci.yml` runs ruff and pytest.

## Next up
Layer 5's retrieval eval (synthetic + hand-labelled queries; recall@10, MRR, nDCG@10 for each mode) should come next. It turns the design choices above into numbers, e.g. bge-small vs minilm vs gemini, and MiniLM vs bge reranker.
