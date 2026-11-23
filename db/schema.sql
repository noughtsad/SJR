-- Idempotent schema. Applied by `python -m app.cli init-db`.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- jobs: one row per (source, external_id). Rows are never hard-deleted by the
-- pipeline; expiry flips is_active so evals and feedback history stay valid.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL,
    external_id     TEXT        NOT NULL,
    url             TEXT,
    title           TEXT        NOT NULL,
    company         TEXT,
    location        TEXT,
    is_remote       BOOLEAN     NOT NULL DEFAULT FALSE,
    seniority       TEXT,
    employment_type TEXT,
    category        TEXT,
    skills          TEXT[]      NOT NULL DEFAULT '{}',
    salary_min      NUMERIC,
    salary_max      NUMERIC,
    salary_currency TEXT,
    description     TEXT        NOT NULL,
    posted_at       TIMESTAMPTZ,

    -- change detection + cross-source dedupe
    content_hash    TEXT        NOT NULL,   -- sha256 of the embeddable content
    dedupe_key      TEXT        NOT NULL,   -- normalised title|company|location

    -- lifecycle
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    expired_at      TIMESTAMPTZ,

    -- keyword search: weighted tsvector, maintained by Postgres
    search_tsv      TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(company, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(category, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'C')
    ) STORED,

    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS jobs_search_tsv_gin ON jobs USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS jobs_skills_gin     ON jobs USING gin (skills);
CREATE INDEX IF NOT EXISTS jobs_active_posted  ON jobs (is_active, posted_at DESC);
CREATE INDEX IF NOT EXISTS jobs_seniority      ON jobs (seniority) WHERE is_active;
CREATE INDEX IF NOT EXISTS jobs_remote         ON jobs (is_remote) WHERE is_active;
CREATE INDEX IF NOT EXISTS jobs_dedupe_key     ON jobs (dedupe_key) WHERE is_active;

-- ---------------------------------------------------------------------------
-- Embeddings live in their own table, keyed by (job, model), so several
-- models can coexist and be compared. The vector column has no fixed
-- dimension; each model gets a *partial expression* HNSW index:
--   CREATE INDEX ... USING hnsw ((embedding::vector(384)) vector_cosine_ops)
--   WHERE model = 'bge-small';
-- (created at runtime by app.embeddings.store.ensure_model_index).
-- content_hash records which version of the job text was embedded, so the
-- pipeline only re-embeds rows whose content actually changed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_models (
    key         TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    dim         INT  NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id        BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    model         TEXT   NOT NULL REFERENCES embedding_models(key),
    content_hash  TEXT   NOT NULL,
    embedding     VECTOR NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, model)
);

-- ---------------------------------------------------------------------------
-- Audit trail for each pipeline run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'running',  -- running | ok | failed
    fetched      INT NOT NULL DEFAULT 0,
    inserted     INT NOT NULL DEFAULT 0,
    updated      INT NOT NULL DEFAULT 0,
    unchanged    INT NOT NULL DEFAULT 0,
    duplicates   INT NOT NULL DEFAULT 0,
    invalid      INT NOT NULL DEFAULT 0,
    expired      INT NOT NULL DEFAULT 0,
    error        TEXT
);

-- ---------------------------------------------------------------------------
-- Document frequencies for keyword search. Postgres ts_rank has no IDF, so a
-- query word like "role" (in most postings) counts as much as "surgery".
-- Refreshed after every ingestion run (app.search.keyword.refresh_lexeme_stats).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lexeme_stats (
    lexeme  TEXT PRIMARY KEY,
    ndoc    INT  NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    id            BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    n_docs        INT NOT NULL,
    refreshed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
