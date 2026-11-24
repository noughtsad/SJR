"""Agent tools. Thin adapters over the search / recommend / profile services.

- The user id is read from the run config, never from tool arguments, so the
  model cannot act on another user's data.
- Every tool returns compact JSON with explicit `job_id` fields. The citation
  check in the graph uses those ids as the set of jobs the agent may mention.
- Argument schemas come from the type hints; invalid arguments come back to
  the model as an error message, and it retries (ToolNode error handling).
"""

import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore
from psycopg_pool import AsyncConnectionPool

from app.agent.memory import FACT_KEYS, put_fact
from app.profile.extract import extract_profile
from app.profile.store import get_profile, save_profile, set_feedback
from app.recommend.explain import explain
from app.recommend.service import RecommendRequest, recommend
from app.search.models import SearchFilters, SearchMode, SearchRequest
from app.search.service import search

DETAIL_CHARS = 3000


def _user(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


def _dump(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def build_tools(pool: AsyncConnectionPool, store: BaseStore) -> list[BaseTool]:
    @tool
    async def search_jobs(
        query: str,
        remote: bool | None = None,
        location: str | None = None,
        seniority: list[Literal["intern", "junior", "mid", "senior", "lead", "executive"]] | None = None,
        k: int = 5,
    ) -> str:
        """Search open jobs by free-text query (role, skills, domain), with optional filters.
        Use for explicit requests like 'find remote rust jobs in Berlin'."""
        req = SearchRequest(
            query=query,
            mode=SearchMode.hybrid_rerank,
            k=max(1, min(k, 10)),
            filters=SearchFilters(remote=remote, location=location, seniority=seniority),
        )
        resp = await search(pool, req)
        return _dump(
            [
                {
                    "job_id": h.job_id,
                    "title": h.title,
                    "company": h.company,
                    "location": h.location,
                    "remote": h.is_remote,
                    "seniority": h.seniority,
                    "snippet": h.snippet[:200],
                }
                for h in resp.results
            ]  # fmt: skip
        )

    @tool
    async def recommend_jobs(config: RunnableConfig, k: int = 5) -> str:
        """Personalised recommendations from the user's resume profile and their saved/dismissed jobs.
        Use when the user asks what fits them. Needs a resume on file."""
        resp = await recommend(pool, RecommendRequest(user_id=_user(config), k=max(1, min(k, 10))))
        return _dump(
            [
                {
                    "job_id": r.job_id,
                    "title": r.title,
                    "company": r.company,
                    "location": r.location,
                    "remote": r.is_remote,
                    "matched_skills": r.matched_skills[:8],
                    "missing_skills": r.missing_skills[:8],
                    "score": r.score,
                }
                for r in resp.results
            ]  # fmt: skip
        )

    @tool
    async def get_job_details(job_id: int) -> str:
        """Full details of one job: description, salary, skills, link."""
        async with pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id AS job_id, title, company, location, is_remote AS remote, seniority, employment_type,
                          skills, salary_min, salary_max, salary_currency, url, posted_at, is_active,
                          left(description, %s) AS description
                   FROM jobs WHERE id = %s""",
                (DETAIL_CHARS, job_id),
            )
            row = await cur.fetchone()
        if not row:
            return _dump({"error": f"no job with id {job_id}"})
        return _dump(row)

    @tool
    async def parse_resume(resume_text: str, config: RunnableConfig) -> str:
        """Extract and save the user's profile from resume text they pasted. Replaces any earlier profile."""
        if len(resume_text) < 100:
            return _dump({"error": "resume text is too short; ask the user to paste the full resume"})
        profile = await extract_profile(resume_text[:20_000])
        version = await save_profile(pool, _user(config), profile, resume_text[:20_000])
        return _dump({"saved": True, "version": version, "profile": profile.model_dump()})

    @tool
    async def skill_gap(job_id: int, config: RunnableConfig) -> str:
        """Compare the user's resume with one job: grounded reasons it fits and requirements they are missing."""
        e = await explain(pool, _user(config), job_id)
        return _dump(
            {
                "job_id": job_id,
                "fit": e.fit,
                "summary": e.summary,
                "reasons": [r.point for r in e.reasons],
                "gaps": [g.requirement for g in e.gaps],
                "matched_skills": e.matched_skills,
                "missing_skills": e.missing_skills,
            }  # fmt: skip
        )

    @tool
    async def save_job(job_id: int, config: RunnableConfig, action: Literal["save", "dismiss"] = "save") -> str:
        """Save a job to the user's list, or dismiss it. Both also tune future recommendations."""
        await set_feedback(pool, _user(config), job_id, action)
        return _dump({"job_id": job_id, "action": action, "ok": True})

    @tool
    async def list_saved_jobs(config: RunnableConfig) -> str:
        """Jobs the user has saved."""
        async with pool.connection() as conn:
            cur = await conn.execute(
                """SELECT j.id AS job_id, j.title, j.company, j.location, j.is_active
                   FROM user_feedback f JOIN jobs j ON j.id = f.job_id
                   WHERE f.user_id = %s AND f.action = 'save' ORDER BY f.created_at DESC LIMIT 20""",
                (_user(config),),
            )
            return _dump(await cur.fetchall())

    @tool
    async def remember_fact(key: FACT_KEYS, value: str, config: RunnableConfig) -> str:
        """Store a durable fact about the user for future conversations (preferences, constraints, goals).
        Only for things they stated about themselves; overwrites the previous value for the key."""
        await put_fact(store, _user(config), key, value)
        return _dump({"remembered": {key: value}})

    @tool
    async def get_my_profile(config: RunnableConfig) -> str:
        """The user's saved resume profile, if any."""
        p = await get_profile(pool, _user(config))
        return _dump(p.model_dump())

    return [
        search_jobs, recommend_jobs, get_job_details, parse_resume, skill_gap,
        save_job, list_saved_jobs, remember_fact, get_my_profile,
    ]  # fmt: skip
