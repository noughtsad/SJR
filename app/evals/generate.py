"""Generate synthetic retrieval queries: sample postings, have the LLM write the
query a job seeker would type to find each one. That posting is the known
relevant result (relevance 2).

  python -m app.evals.generate --n 80 [--seed 7]

Caveats, stated up front because interviewers will ask:
- Only the source posting counts as relevant. Other genuinely matching jobs
  in the index score as misses, so absolute recall is a lower bound. That's
  fine for comparing modes, which all face the same handicap.
- LLM-written queries tend to echo the posting's wording, which flatters
  keyword search. The prompt forbids copying the title or company, and the
  'natural' and 'skills' styles push toward paraphrase.
"""

import argparse
import asyncio
import random

from pydantic import BaseModel, Field

from app.db import make_pool, run
from app.evals.dataset import SYNTHETIC_PATH, EvalQuery, job_key, load, save
from app.llm import generate_structured

STYLES = {
    "keyword": "a short search-box query of 2-5 words, like people type into a job site",
    "natural": "one natural sentence describing what the job seeker wants (role, level, setting), 10-25 words",
    "skills": "a query built around the candidate's skills/experience rather than a job title, 5-15 words",
}

SYSTEM = """You write realistic search queries that job seekers type into a job search engine.
Rules:
- Never include the company name.
- Do not copy the job title verbatim; paraphrase or describe the role the way a candidate would.
- The query must be specific enough that the given posting is a strong match, but realistic:
  a candidate does not know the posting's exact wording.
- Use only facts present in the posting (location / remote / seniority only if the posting states them)."""

# Sample proportionally-ish, but make sure the small live sources are represented.
SOURCE_QUOTAS = {"kaggle_linkedin": 0.7, "hn_hiring": 0.25, "remotive": 0.05}


class _Item(BaseModel):
    index: int = Field(description="the [n] index of the posting this query is for")
    query: str


class _Batch(BaseModel):
    queries: list[_Item]


async def _sample(pool, n: int, seed: int, exclude: set[str]) -> list[dict]:
    rows: list[dict] = []
    async with pool.connection() as conn:
        await conn.execute("SELECT setseed(%s)", (seed / 1000,))
        for source, share in SOURCE_QUOTAS.items():
            cur = await conn.execute(
                """SELECT source, external_id, title, company, location, is_remote, seniority,
                          left(description, 1500) AS description
                   FROM jobs WHERE is_active AND source = %s AND length(description) >= 300
                   ORDER BY random() LIMIT %s""",
                (source, max(1, round(n * share)) + 20),
            )
            got = [r for r in await cur.fetchall() if job_key(r["source"], r["external_id"]) not in exclude]
            rows += got[: max(1, round(n * share))]
    return rows[:n]


def _prompt(jobs: list[dict], styles: list[str]) -> str:
    parts = ["Write one query for each posting below, in the requested style.\n"]
    for i, (j, style) in enumerate(zip(jobs, styles, strict=True)):
        parts.append(
            f"[{i}] STYLE: {style} — {STYLES[style]}\n"
            f"Title: {j['title']}\nCompany: {j['company']}\nLocation: {j['location']}"
            f"{' (remote)' if j['is_remote'] else ''}\nSeniority: {j['seniority'] or 'unspecified'}\n"
            f"Description: {j['description']}\n"
        )
    return "\n".join(parts)


async def generate(n: int, seed: int, batch_size: int = 10) -> list[EvalQuery]:
    existing = load(SYNTHETIC_PATH)
    taken = {k for q in existing for k in q.qrels}
    pool = make_pool()
    await pool.open()
    try:
        jobs = await _sample(pool, n, seed, taken)
    finally:
        await pool.close()

    rng = random.Random(seed)
    style_names = list(STYLES)
    out: list[EvalQuery] = []
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b : b + batch_size]
        styles = [style_names[(b + i) % 3] for i in range(len(chunk))]
        rng.shuffle(styles)
        resp = await generate_structured(_prompt(chunk, styles), _Batch, system=SYSTEM, temperature=0.8)
        by_index = {it.index: it.query.strip() for it in resp.queries}
        for i, (j, style) in enumerate(zip(chunk, styles, strict=True)):
            q = by_index.get(i)
            if not q:
                continue  # model skipped one; drop rather than guess
            out.append(
                EvalQuery(
                    qid=f"syn-{len(existing) + len(out) + 1:03d}",
                    query=q,
                    set="synthetic",
                    style=style,
                    qrels={job_key(j["source"], j["external_id"]): 2},
                )
            )
        print(f"generated {len(out)}/{len(jobs)}")
        await asyncio.sleep(1)  # be gentle with free-tier rate limits
    save(SYNTHETIC_PATH, existing + out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    qs = run(generate(a.n, a.seed))
    print(f"wrote {len(qs)} queries to {SYNTHETIC_PATH}")
    for q in qs[:8]:
        print(f"  [{q.style}] {q.query}")


if __name__ == "__main__":
    main()
