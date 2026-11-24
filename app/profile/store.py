"""Persistence for users, profiles and feedback."""

from typing import Literal

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.profile.extract import canonical_skills
from app.profile.models import CandidateProfile, StoredProfile

FeedbackAction = Literal["save", "dismiss"]


class NotFound(LookupError):
    pass


async def save_profile(pool: AsyncConnectionPool, user_id: str, profile: CandidateProfile, resume_text: str) -> int:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
        cur = await conn.execute(
            """INSERT INTO user_profiles (user_id, profile, resume_text) VALUES (%s, %s, %s)
               ON CONFLICT (user_id) DO UPDATE
               SET profile = EXCLUDED.profile, resume_text = EXCLUDED.resume_text,
                   version = user_profiles.version + 1, updated_at = now()
               RETURNING version""",
            (user_id, Jsonb(profile.model_dump()), resume_text),
        )
        return (await cur.fetchone())["version"]


async def get_profile(pool: AsyncConnectionPool, user_id: str) -> StoredProfile:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT profile, version FROM user_profiles WHERE user_id = %s", (user_id,))
        row = await cur.fetchone()
    if not row:
        raise NotFound(f"no profile for user {user_id!r}; upload a resume first")
    p = CandidateProfile.model_validate(row["profile"])
    return StoredProfile(user_id=user_id, profile=p, canonical_skills=canonical_skills(p), version=row["version"])


async def set_feedback(pool: AsyncConnectionPool, user_id: str, job_id: int, action: FeedbackAction | None) -> None:
    """Record the latest signal for (user, job); action=None clears it."""
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute("SELECT 1 FROM jobs WHERE id = %s", (job_id,))
        if not await cur.fetchone():
            raise NotFound(f"no job {job_id}")
        await conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
        if action is None:
            await conn.execute("DELETE FROM user_feedback WHERE user_id = %s AND job_id = %s", (user_id, job_id))
        else:
            await conn.execute(
                """INSERT INTO user_feedback (user_id, job_id, action) VALUES (%s, %s, %s)
                   ON CONFLICT (user_id, job_id) DO UPDATE SET action = EXCLUDED.action, created_at = now()""",
                (user_id, job_id, action),
            )


async def get_feedback(pool: AsyncConnectionPool, user_id: str) -> dict[int, str]:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT job_id, action FROM user_feedback WHERE user_id = %s", (user_id,))
        return {r["job_id"]: r["action"] for r in await cur.fetchall()}
