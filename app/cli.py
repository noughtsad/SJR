"""Command line entry point.

  python -m app.cli init-db
  python -m app.cli ingest remotive [--from-file data/raw/remotive_x.json]
  python -m app.cli ingest hn [--months 2]
  python -m app.cli ingest kaggle --path data/postings.csv --limit 5000
  python -m app.cli embed [--model bge-small] [--limit N]
  python -m app.cli refresh            # ingest live sources + embed (what the cron runs)
  python -m app.cli search "python backend remote" [--mode hybrid] [--remote]
  python -m app.cli stats
"""

import argparse
import json
import logging
from pathlib import Path

from app.db import init_db, make_pool, run
from app.embeddings.store import embed_pending
from app.ingestion.pipeline import run_ingestion
from app.search.models import SearchFilters, SearchMode, SearchRequest

log = logging.getLogger("app.cli")


def _source(args):
    if args.source == "remotive":
        from app.ingestion.sources.remotive import RemotiveSource

        return RemotiveSource(from_file=Path(args.from_file) if args.from_file else None)
    if args.source == "hn":
        from app.ingestion.sources.hn_hiring import HNHiringSource

        return HNHiringSource(months=args.months)
    from app.ingestion.sources.kaggle_csv import KaggleCsvSource

    if not args.path:
        raise SystemExit("--path is required for the kaggle source")
    return KaggleCsvSource(Path(args.path), limit=args.limit)


async def _with_pool(fn):
    pool = make_pool()
    await pool.open()
    try:
        return await fn(pool)
    finally:
        await pool.close()


async def cmd_ingest(args):
    stats = await _with_pool(lambda p: run_ingestion(p, _source(args)))
    print(json.dumps(stats.__dict__, indent=2))


async def cmd_embed(args):
    n = await _with_pool(lambda p: embed_pending(p, args.model, args.limit))
    print(f"embedded {n} jobs")


async def cmd_refresh(args):
    from app.ingestion.sources.hn_hiring import HNHiringSource
    from app.ingestion.sources.remotive import RemotiveSource

    async def go(pool):
        failures = 0
        for src in (RemotiveSource(), HNHiringSource()):
            try:  # one broken source shouldn't block the others
                stats = await run_ingestion(pool, src)
                print(f"ingest {src.name}:", json.dumps(stats.__dict__))
            except Exception:
                failures += 1
                log.exception("ingestion failed for %s", src.name)
        print("embedded:", await embed_pending(pool, args.model))
        if failures:
            raise SystemExit(1)

    await _with_pool(go)


async def cmd_search(args):
    from app.search.service import search

    req = SearchRequest(
        query=args.query,
        mode=SearchMode(args.mode),
        k=args.k,
        model=args.model,
        filters=SearchFilters(remote=True if args.remote else None, location=args.location, seniority=args.seniority),
    )
    resp = await _with_pool(lambda p: search(p, req))
    print(f"mode={resp.mode} model={resp.model} timings_ms={resp.timings_ms}\n")
    for i, h in enumerate(resp.results, 1):
        sig = "  ".join(
            f"{name}={val:.3f}" if isinstance(val, float) else f"{name}={val}"
            for name, val in (("kw#", h.keyword_rank), ("vec#", h.vector_rank),
                              ("rrf", h.rrf_score), ("rerank", h.rerank_score))
            if val is not None
        )  # fmt: skip
        print(f"{i:>2}. {h.title} | {h.company} | {h.location} [{h.seniority or '-'}]\n    {sig}")


async def cmd_stats(_args):
    async def go(pool):
        async with pool.connection() as conn:
            for q in (
                "SELECT source, count(*) FILTER (WHERE is_active) AS active, count(*) AS total FROM jobs GROUP BY 1",
                "SELECT model, count(*) AS n FROM job_embeddings GROUP BY 1",
                "SELECT id, source, status, fetched, inserted, updated, unchanged, duplicates, invalid, expired,"
                " finished_at - started_at AS took FROM ingestion_runs ORDER BY id DESC LIMIT 5",
            ):
                cur = await conn.execute(q)
                for r in await cur.fetchall():
                    print(dict(r))
                print()

    await _with_pool(go)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")

    pi = sub.add_parser("ingest")
    pi.add_argument("source", choices=["remotive", "hn", "kaggle"])
    pi.add_argument("--from-file")
    pi.add_argument("--months", type=int, default=2, help="hn: how many monthly threads count as open")
    pi.add_argument("--path")
    pi.add_argument("--limit", type=int)

    pe = sub.add_parser("embed")
    pe.add_argument("--model")
    pe.add_argument("--limit", type=int)

    pr = sub.add_parser("refresh")
    pr.add_argument("--model")

    ps = sub.add_parser("search")
    ps.add_argument("query")
    ps.add_argument("--mode", default="hybrid_rerank", choices=[m.value for m in SearchMode])
    ps.add_argument("--k", type=int, default=10)
    ps.add_argument("--model")
    ps.add_argument("--remote", action="store_true")
    ps.add_argument("--location")
    ps.add_argument("--seniority", nargs="*")

    sub.add_parser("stats")

    args = p.parse_args()
    if args.cmd == "init-db":
        run(init_db())
        print("schema applied")
        return
    handler = {
        "ingest": cmd_ingest, "embed": cmd_embed, "refresh": cmd_refresh,
        "search": cmd_search, "stats": cmd_stats,
    }[args.cmd]  # fmt: skip
    run(handler(args))


if __name__ == "__main__":
    main()
