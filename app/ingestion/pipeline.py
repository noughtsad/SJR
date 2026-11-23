"""Fetch -> validate -> dedupe -> diff against DB -> upsert -> expire.

Each posting is classified against the stored row for (source, external_id):
  new        -> INSERT
  changed    -> UPDATE content (content_hash differs; the embedder picks it up)
  unchanged  -> touch last_seen_at only
Postings whose dedupe_key already belongs to a *different* active job
(same title/company/location from another source or id) are counted as
duplicates and skipped.
"""

import logging
from dataclasses import asdict, dataclass

from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from app.config import get_settings
from app.ingestion.models import JobPosting
from app.ingestion.normalize import content_hash, dedupe_key, looks_like_self_promotion
from app.ingestion.sources.base import JobSource
from app.search.keyword import refresh_lexeme_stats

log = logging.getLogger(__name__)

BATCH = 500

_JOB_COLS = [
    "source", "external_id", "url", "title", "company", "location", "is_remote",
    "seniority", "employment_type", "category", "skills", "salary_min", "salary_max",
    "salary_currency", "description", "posted_at", "content_hash", "dedupe_key",
]  # fmt: skip


@dataclass
class RunStats:
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicates: int = 0
    invalid: int = 0
    rejected: int = 0
    expired: int = 0


def _row(job: JobPosting) -> dict:
    d = job.model_dump()
    d["content_hash"] = content_hash(job)
    d["dedupe_key"] = dedupe_key(job)
    return d


async def _apply_batch(conn, source: str, rows: list[dict], stats: RunStats) -> None:
    ids = [r["external_id"] for r in rows]
    cur = await conn.execute(
        "SELECT external_id, content_hash FROM jobs WHERE source = %s AND external_id = ANY(%s)",
        (source, ids),
    )
    existing = {r["external_id"]: r["content_hash"] for r in await cur.fetchall()}

    # Cross-source dedupe: is this dedupe_key already owned by another active job?
    keys = list({r["dedupe_key"] for r in rows})
    cur = await conn.execute(
        "SELECT dedupe_key, source, external_id FROM jobs WHERE is_active AND dedupe_key = ANY(%s)",
        (keys,),
    )
    owners = {r["dedupe_key"]: (r["source"], r["external_id"]) for r in await cur.fetchall()}

    new, changed, unchanged = [], [], []
    for r in rows:
        if r["external_id"] in existing:
            (unchanged if existing[r["external_id"]] == r["content_hash"] else changed).append(r)
            continue
        owner = owners.get(r["dedupe_key"])
        if owner and owner != (source, r["external_id"]):
            stats.duplicates += 1
            continue
        owners[r["dedupe_key"]] = (source, r["external_id"])
        new.append(r)

    async with conn.transaction():
        if new:
            cols = ", ".join(_JOB_COLS)
            vals = ", ".join(f"%({c})s" for c in _JOB_COLS)
            async with conn.cursor() as c:
                await c.executemany(
                    f"INSERT INTO jobs ({cols}) VALUES ({vals}) ON CONFLICT (source, external_id) DO NOTHING",
                    new,
                )
        if changed:
            sets = ", ".join(f"{c} = %({c})s" for c in _JOB_COLS if c not in ("source", "external_id"))
            async with conn.cursor() as c:
                await c.executemany(
                    f"""UPDATE jobs SET {sets}, updated_at = now(), last_seen_at = now(),
                               is_active = TRUE, expired_at = NULL
                        WHERE source = %(source)s AND external_id = %(external_id)s""",
                    changed,
                )
        if unchanged:
            await conn.execute(
                """UPDATE jobs SET last_seen_at = now(), is_active = TRUE, expired_at = NULL
                   WHERE source = %s AND external_id = ANY(%s)""",
                (source, [r["external_id"] for r in unchanged]),
            )

    stats.inserted += len(new)
    stats.updated += len(changed)
    stats.unchanged += len(unchanged)


async def _expire(conn, source: JobSource, run_started, stats: RunStats) -> None:
    s = get_settings()
    if source.full_snapshot:
        cur = await conn.execute("SELECT count(*) AS n FROM jobs WHERE source = %s AND is_active", (source.name,))
        active = (await cur.fetchone())["n"]
        seen = stats.inserted + stats.updated + stats.unchanged
        if active and seen < s.min_snapshot_ratio * active:
            log.warning(
                "skip expiry for %s: saw %d of %d active jobs (below %.0f%%), response looks partial",
                source.name, seen, active, s.min_snapshot_ratio * 100,
            )  # fmt: skip
            return
        cur = await conn.execute(
            """UPDATE jobs SET is_active = FALSE, expired_at = now()
               WHERE source = %s AND is_active AND last_seen_at < %s""",
            (source.name, run_started),
        )
        stats.expired += cur.rowcount
    elif source.ttl_days:
        cur = await conn.execute(
            """UPDATE jobs SET is_active = FALSE, expired_at = now()
               WHERE source = %s AND is_active
                 AND coalesce(posted_at, first_seen_at) < now() - make_interval(days => %s)""",
            (source.name, source.ttl_days),
        )
        stats.expired += cur.rowcount


async def run_ingestion(pool: AsyncConnectionPool, source: JobSource) -> RunStats:
    stats = RunStats()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO ingestion_runs (source) VALUES (%s) RETURNING id, started_at", (source.name,)
        )
        run = await cur.fetchone()

    seen_ids: set[str] = set()
    batch: list[dict] = []
    status, error = "failed", None
    try:
        async with pool.connection() as conn:
            async for raw in source.fetch_raw():
                stats.fetched += 1
                try:
                    job = source.to_posting(raw)
                except (ValidationError, KeyError, ValueError) as e:
                    stats.invalid += 1
                    log.debug("invalid record from %s: %s", source.name, e)
                    continue
                if source.user_generated and looks_like_self_promotion(job.description):
                    stats.rejected += 1  # not a job ad; never inserted, so any old copy expires
                    log.info("rejected self-promotion post %s:%s", source.name, job.external_id)
                    continue
                if job.external_id in seen_ids:  # duplicate id inside one feed
                    stats.duplicates += 1
                    continue
                seen_ids.add(job.external_id)
                batch.append(_row(job))
                if len(batch) >= BATCH:
                    await _apply_batch(conn, source.name, batch, stats)
                    batch.clear()
            if batch:
                await _apply_batch(conn, source.name, batch, stats)
            await _expire(conn, source, run["started_at"], stats)
            if stats.inserted or stats.updated or stats.expired:
                await refresh_lexeme_stats(conn)  # keyword IDF must track the corpus
            status = "ok"
    except Exception as e:
        error = repr(e)
        raise
    finally:
        counters = ", ".join(f"{k} = %({k})s" for k in asdict(stats))
        async with pool.connection() as conn:
            await conn.execute(
                f"""UPDATE ingestion_runs
                    SET finished_at = now(), status = %(status)s, error = %(error)s, {counters}
                    WHERE id = %(id)s""",
                {**asdict(stats), "status": status, "error": error, "id": run["id"]},
            )
    return stats
