"""Async Postgres access: a psycopg3 connection pool with pgvector types registered."""

import asyncio
import sys
from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def loop_factory():
    """psycopg's async driver cannot use Windows' default Proactor loop."""
    return asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()


def run(coro):
    return asyncio.run(coro, loop_factory=loop_factory)


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)
    conn.row_factory = dict_row


async def _configure_bootstrap(conn: AsyncConnection) -> None:
    # Used before the `vector` extension exists (first init-db run).
    conn.row_factory = dict_row


def make_pool(*, bootstrap: bool = False, statement_timeout_ms: int | None = None) -> AsyncConnectionPool:
    s = get_settings()
    kwargs: dict = {"autocommit": True}
    if statement_timeout_ms:
        kwargs["options"] = f"-c statement_timeout={int(statement_timeout_ms)}"
    return AsyncConnectionPool(
        s.database_url,
        min_size=s.db_pool_min,
        max_size=s.db_pool_max,
        configure=_configure_bootstrap if bootstrap else _configure,
        kwargs=kwargs,
        open=False,
    )


async def init_db() -> None:
    pool = make_pool(bootstrap=True)
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        await pool.close()
