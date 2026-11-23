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
- *Keyword (BM25-lite on Postgres FTS)*: Postgres `ts_rank` has **no IDF**. ANDing all terms matches nothing for natural queries. ORing them let words like "want" and "role" match 15.6k of 19.4k jobs, and ranking those took ~700 ms. The fix: a `lexeme_stats` table holds document frequencies (via `ts_stat`, ~1.6 s, refreshed after each ingest). Query terms found in more than 15% of jobs are dropped, and candidates are scored by Σ BM25-IDF of matched terms, with title matches counting double. Full BM25's term-frequency saturation and length normalisation are not modelled.
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

## Retrieval evals

```bash
python -m app.evals.generate --n 80      # synthetic queries (Gemini), appended to evals/retrieval/synthetic.jsonl
python -m app.evals.label                # hand-label the 30 realistic queries (interactive)
python -m app.evals.run                  # all modes; table -> evals/results/latest.md
python -m app.evals.run --models bge-small minilm gemini   # compare embedding models
```

**Two query sets**
- **Synthetic (80)**: sampled postings (70% LinkedIn, 25% HN, 5% Remotive). Gemini writes the query a job seeker would type, in three styles: `keyword` (2–5 words), `natural` (a sentence), `skills` (built around experience, not the title). The source posting is the single known-relevant result. The prompt bans the company name and verbatim titles. An audit found no company leaks; 18/80 contain the title, almost all generic ones that real users type ("Project Manager").
- **AI-judged (30 queries, 607 judgements)**: realistic queries written by hand, judged with TREC-style **pooling**. The top 10 from all four modes are merged and graded 0/1/2 by Claude against a written rubric ([`JUDGING_RUBRIC.md`](evals/retrieval/JUDGING_RUBRIC.md)), without knowing which mode returned which job. This set is *not* human-labelled. A human set can be added with `python -m app.evals.label`, which pools the same way, pre-suggests grades and has a person confirm each one. It is stored separately in `handlabeled.jsonl`.

**Metrics**: Recall@10, MRR@10, nDCG@10 (graded, exponential gain), plus p50/p95 latency.

**Known biases, stated deliberately**
- Synthetic: only the source posting counts as relevant, so other genuinely matching jobs score as misses. Absolute recall is a lower bound. Comparisons between modes are still fair because every mode faces the same handicap.
- Judged set: unjudged documents count as non-relevant (the standard pooling assumption), so recall means "recall within the judged pool". Most queries have more than 10 relevant jobs, so Recall@10 can't reach 1; read it comparatively. Judge strictness varied between the three judges (see the rubric file). That affects absolute scores, but not comparisons between modes, since every mode is scored against the same labels.
- Labels key on `source:external_id`, not the serial `jobs.id`, so the eval set survives a database rebuild. Queries whose target job has since expired are skipped and counted.

**Results** (80 synthetic queries, ~19.4k jobs, bge-small, RTX 3070; full table in [`evals/results/latest.md`](evals/results/latest.md)):

| Mode | Recall@10 | MRR@10 | nDCG@10 | p50 ms |
|---|---|---|---|---|
| keyword | 0.738 | 0.483 | 0.545 | 53 |
| vector | 0.787 | 0.702 | 0.723 | 13 |
| hybrid (RRF) | 0.838 | 0.670 | 0.711 | 56 |
| **hybrid + rerank** | **0.912** | **0.841** | **0.858** | 107 |

**What the eval caught.** The first run had keyword search at recall 0.30 / MRR 0.13 and 200 ms. Equal-weight RRF then made hybrid *worse* than vector alone (MRR 0.40 vs 0.70). Profiling found the missing-IDF problem above. After the fix, keyword search improved 2.5× in recall and 4× in speed, and hybrid now beats vector on recall (0.84 vs 0.79).

**Tuning, and resisting over-tuning.** A sweep of the DF cutoff (0.05–0.4) showed 0.25 gives better keyword-only MRR, but no gain after fusion at twice the latency, so 0.15 stays. Weighting vector 2–3× in RRF helps plain hybrid slightly but hurts hybrid+rerank, the production mode, because the reranker benefits from a more diverse top 30. Weights stay 1:1. With 80 queries, differences of about 0.03 are 2–3 queries, so they are not tuned further until the hand-labelled set exists.

**On realistic queries (AI-judged set, 30 queries):**

| Mode | Recall@10 | MRR@10 | nDCG@10 | p50 ms |
|---|---|---|---|---|
| keyword | 0.416 | 0.796 | 0.597 | 15 |
| vector | **0.604** | 0.978 | 0.803 | 11 |
| hybrid (RRF) | 0.555 | 0.967 | 0.784 | 14 |
| **hybrid + rerank** | 0.590 | **0.983** | **0.820** | 49 |

- **MRR saturates.** Realistic queries ("HR generalist") have many matching jobs, so every semantic mode nearly always puts a relevant job first. nDCG and recall are the measures that separate the modes here.
- **Hybrid+rerank still ranks best (nDCG 0.820), but vector alone is close (0.803) at a quarter of the latency.** That is a real product trade-off. A latency-sensitive endpoint, such as autocomplete or the agent's internal `search_jobs` calls, could use vector-only.
- **On short realistic queries, equal-weight RRF still lets keyword search pull hybrid below vector.** This matches the synthetic sweep, where weighting vector more helped plain hybrid. With 30 queries, differences of about 0.02 are within noise.

GPU vs CPU (i7-12650H): embedding 417 vs ~7 docs/s, rerank of 30 docs 58 ms vs ~1 s.

## Next up
Layer 3: resume → profile, feedback-driven personalisation, grounded match explanations.
