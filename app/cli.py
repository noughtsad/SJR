"""Command line entry point.

  python -m app.cli init-db
  python -m app.cli ingest remotive [--from-file data/raw/remotive_x.json]
  python -m app.cli ingest hn [--months 2]
  python -m app.cli ingest kaggle --path data/postings.csv --limit 5000
  python -m app.cli embed [--model bge-small] [--limit N]
  python -m app.cli refresh            # ingest live sources + embed (what the cron runs)
  python -m app.cli search "python backend remote" [--mode hybrid] [--remote]
  python -m app.cli stats
  python -m app.cli profile --user demo --resume examples/resume_data_engineer.md
  python -m app.cli recommend --user demo [--k 10] [--no-feedback]
  python -m app.cli feedback --user demo --job 123 --action save|dismiss|clear
  python -m app.cli explain --user demo --job 123 [--refresh]
  python -m app.cli chat --user demo [--thread t1] [-m "one message"]
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


async def cmd_refresh_stats(_args):
    from app.search.keyword import refresh_lexeme_stats

    async def go(pool):
        async with pool.connection() as conn:
            print(f"lexeme_stats: {await refresh_lexeme_stats(conn)} lexemes")

    await _with_pool(go)


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


async def cmd_profile(args):
    from app.profile.extract import extract_profile, read_resume_file
    from app.profile.store import get_profile, save_profile

    text = read_resume_file(Path(args.resume))

    async def go(pool):
        profile = await extract_profile(text)
        await save_profile(pool, args.user, profile, text)
        return await get_profile(pool, args.user)

    stored = await _with_pool(go)
    print(json.dumps(stored.model_dump(), indent=2))


async def cmd_recommend(args):
    from app.recommend.service import RecommendRequest, recommend

    req = RecommendRequest(user_id=args.user, k=args.k, use_feedback=not args.no_feedback)
    resp = await _with_pool(lambda p: recommend(p, req))
    print(f"profile v{resp.profile_version}  feedback={resp.feedback_used}  "
          f"query_shift={resp.query_shift}  timings_ms={resp.timings_ms}\n")  # fmt: skip
    for i, r in enumerate(resp.results, 1):
        cov = "-" if r.skill_coverage is None else f"{r.skill_coverage:.0%}"
        print(f"{i:>2}. [{r.job_id}] {r.title} | {r.company} | {r.location} [{r.seniority or '-'}]")
        print(f"    score={r.score:.3f} sim={r.similarity:.3f} skills={cov} gap={r.seniority_gap}"
              f"  has={r.matched_skills[:6]} missing={r.missing_skills[:6]}")  # fmt: skip


async def cmd_feedback(args):
    from app.profile.store import set_feedback

    action = None if args.action == "clear" else args.action
    await _with_pool(lambda p: set_feedback(p, args.user, args.job, action))
    print(f"{args.user}: job {args.job} -> {args.action}")


async def cmd_explain(args):
    from app.recommend.explain import explain

    e = await _with_pool(lambda p: explain(p, args.user, args.job, refresh=args.refresh))
    print(json.dumps(e.model_dump(), indent=2))


async def cmd_chat(args):
    import asyncio
    import uuid

    from app.agent.runtime import agent_runtime, chat_stream

    thread = args.thread or uuid.uuid4().hex[:8]

    async def turn(graph, text):
        async for ev in chat_stream(graph, args.user, thread, text):
            t = ev["type"]
            if t == "token":
                print(ev["text"], end="", flush=True)
            elif t == "tool_call":
                print(f"\n  -> {ev['name']}({json.dumps(ev['args'], ensure_ascii=False)})", flush=True)
            elif t == "tool_result":
                print(f"  <- {ev['name']}: {ev['preview'][:120]}...", flush=True)
            elif t == "retract":
                print("\n  !! draft rejected by citation check, regenerating", flush=True)
            elif t == "error":
                print(f"\n  !! {ev['kind']}: {ev['message']}\n", flush=True)
            elif t == "final":
                print(f"\n\n[cited={ev['cited_job_ids']} usage={ev['usage']} {ev['latency_ms']} ms]\n")

    async def go(pool):
        async with agent_runtime(pool) as (graph, _store):
            print(f"thread {thread}  (user {args.user}); empty line or Ctrl+C to quit\n")
            if args.message:
                await turn(graph, args.message)
                return
            while True:
                text = (await asyncio.to_thread(input, "you> ")).strip()
                if not text:
                    return
                await turn(graph, text)

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
    sub.add_parser("refresh-stats", help="recompute keyword-search document frequencies")

    pp = sub.add_parser("profile", help="extract a profile from a resume (.pdf/.md/.txt)")
    pp.add_argument("--user", required=True)
    pp.add_argument("--resume", required=True)

    prec = sub.add_parser("recommend")
    prec.add_argument("--user", required=True)
    prec.add_argument("--k", type=int, default=10)
    prec.add_argument("--no-feedback", action="store_true")

    pf = sub.add_parser("feedback")
    pf.add_argument("--user", required=True)
    pf.add_argument("--job", type=int, required=True)
    pf.add_argument("--action", choices=["save", "dismiss", "clear"], required=True)

    px = sub.add_parser("explain")
    px.add_argument("--user", required=True)
    px.add_argument("--job", type=int, required=True)
    px.add_argument("--refresh", action="store_true")

    pc = sub.add_parser("chat", help="talk to the career agent (streams tokens and tool calls)")
    pc.add_argument("--user", required=True)
    pc.add_argument("--thread", help="continue a conversation; default starts a new one")
    pc.add_argument("-m", "--message", help="send one message and exit")

    args = p.parse_args()
    if args.cmd == "init-db":
        run(init_db())
        print("schema applied")
        return
    handler = {
        "ingest": cmd_ingest, "embed": cmd_embed, "refresh": cmd_refresh,
        "search": cmd_search, "stats": cmd_stats, "refresh-stats": cmd_refresh_stats,
        "profile": cmd_profile, "recommend": cmd_recommend, "feedback": cmd_feedback, "explain": cmd_explain,
        "chat": cmd_chat,
    }[args.cmd]  # fmt: skip
    run(handler(args))


if __name__ == "__main__":
    main()
