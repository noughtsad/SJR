"""CI corpus: a small, fixed slice of the index that ships with the repo.

  python -m app.evals.fixture build [--distractors 1500]   # from the full local DB
  python -m app.evals.fixture load                         # into an empty DB (CI)

The same format moves the whole index to a hosted DB (e.g. Supabase) without re-embedding:
  python -m app.evals.fixture build --all --out data/full_index.jsonl.gz
  DATABASE_URL=<hosted> python -m app.evals.fixture load --path data/full_index.jsonl.gz

It contains every job referenced by the retrieval eval labels, plus a seeded
random sample of other jobs as distractors, with their bge-small embeddings
(float16), so CI needs neither the Kaggle dump nor a GPU. Metrics on this
corpus are not comparable to the full-index numbers in the README (the
haystack is ~10x smaller); they are a regression baseline. CI fails when a
change drops them below evals/ci_thresholds.json.
"""

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np
from psycopg import sql

from app.db import init_db, make_pool, run
from app.embeddings.registry import get_spec
from app.embeddings.store import ensure_model_index
from app.evals.dataset import HANDLABELED_PATH, JUDGED_PATH, SYNTHETIC_PATH, load
from app.search.keyword import refresh_lexeme_stats

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "ci_corpus.jsonl.gz"
DESC_CHARS = 6000
COLS = [
    "source", "external_id", "url", "title", "company", "location", "is_remote", "seniority",
    "employment_type", "category", "skills", "salary_min", "salary_max", "salary_currency",
    "description", "posted_at", "content_hash", "dedupe_key",
]  # fmt: skip


async def build(distractors: int, model_key: str, out: Path = FIXTURE_PATH) -> None:
    spec = get_spec(model_key)
    keys = {k for path in (SYNTHETIC_PATH, JUDGED_PATH, HANDLABELED_PATH) for q in load(path) for k in q.qrels}
    pairs = [k.split(":", 1) for k in keys]
    pool = make_pool()
    await pool.open()
    try:
        async with pool.connection() as conn:
            q = sql.SQL("""
                WITH labelled AS (
                    SELECT j.id FROM jobs j
                    JOIN unnest(%(src)s::text[], %(ext)s::text[]) AS k(source, external_id)
                      ON j.source = k.source AND j.external_id = k.external_id
                ), sample AS (  -- deterministic "random" sample: order by a hash of the id
                    SELECT id FROM jobs WHERE is_active AND id NOT IN (SELECT id FROM labelled)
                    ORDER BY md5(id::text) LIMIT %(n)s
                )
                SELECT j.*, e.embedding::vector({dim}) AS emb
                FROM jobs j JOIN job_embeddings e ON e.job_id = j.id AND e.model = {model}
                WHERE j.id IN (SELECT id FROM labelled UNION ALL SELECT id FROM sample)
                ORDER BY j.source, j.external_id
            """).format(dim=sql.Literal(spec.dim), model=sql.Literal(spec.key))
            cur = await conn.execute(q, {"src": [p[0] for p in pairs], "ext": [p[1] for p in pairs], "n": distractors})
            rows = await cur.fetchall()
    finally:
        await pool.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"model": spec.key, "dim": spec.dim}) + "\n")
        for r in rows:
            d = {c: r[c] for c in COLS}
            d["description"] = d["description"][:DESC_CHARS]
            d["embedding"] = base64.b64encode(r["emb"].to_numpy().astype(np.float16).tobytes()).decode()
            f.write(json.dumps(d, default=str) + "\n")
    found = sum(1 for r in rows if f"{r['source']}:{r['external_id']}" in keys)
    print(f"wrote {len(rows)} jobs ({found}/{len(keys)} labelled, rest distractors) -> {out}")
    print(f"size: {out.stat().st_size / 1e6:.1f} MB")


async def load_fixture(path: Path = FIXTURE_PATH) -> None:
    await init_db()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    spec = get_spec(header["model"])
    pool = make_pool()
    await pool.open()
    try:
        await ensure_model_index(pool, spec)  # registers the model + builds its HNSW index
        async with pool.connection() as conn, conn.transaction():
            cols = ", ".join(COLS)
            vals = ", ".join(f"%({c})s" for c in COLS)
            async with conn.cursor() as cur:
                await cur.executemany(
                    f"INSERT INTO jobs ({cols}) VALUES ({vals}) ON CONFLICT (source, external_id) DO NOTHING", rows
                )
            cur = await conn.execute("SELECT id, source, external_id, content_hash FROM jobs")
            ids = {(r["source"], r["external_id"]): (r["id"], r["content_hash"]) for r in await cur.fetchall()}
            emb_rows = []
            for r in rows:
                job_id, chash = ids[(r["source"], r["external_id"])]
                v = np.frombuffer(base64.b64decode(r["embedding"]), dtype=np.float16).astype(np.float32)
                emb_rows.append((job_id, spec.key, chash, v))
            async with conn.cursor() as cur:
                await cur.executemany(
                    """INSERT INTO job_embeddings (job_id, model, content_hash, embedding) VALUES (%s, %s, %s, %s)
                       ON CONFLICT (job_id, model) DO NOTHING""",
                    emb_rows,
                )
        async with pool.connection() as conn:
            n = await refresh_lexeme_stats(conn)
    finally:
        await pool.close()
    print(f"loaded {len(rows)} jobs + {spec.key} embeddings; lexeme_stats: {n}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--distractors", type=int, default=1500)
    b.add_argument("--model", default="bge-small")
    b.add_argument("--all", action="store_true", help="every active job (for moving the index to a hosted DB)")
    b.add_argument("--out", type=Path, default=FIXTURE_PATH)
    ld = sub.add_parser("load")
    ld.add_argument("--path", type=Path, default=FIXTURE_PATH)
    a = p.parse_args()
    if a.cmd == "build":
        run(build(10**9 if a.all else a.distractors, a.model, a.out))
    else:
        run(load_fixture(a.path))


if __name__ == "__main__":
    main()
