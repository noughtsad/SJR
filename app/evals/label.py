"""Interactive relevance labelling for the hand-written query set (TREC-style pooling).

  python -m app.evals.label            # label every query that has no judgements yet
  python -m app.evals.label --redo hl-007

For each query, the top-10 results of keyword, vector, hybrid and hybrid+rerank are
pooled, de-duplicated and *shuffled* (so you can't tell which system found what),
then you grade each job:
    2 = great match   1 = partial / acceptable   0 = not relevant
    s = skip query    q = save and quit
With assist on (default), Gemini pre-grades the pool and Enter accepts its grade.
You still see and confirm every judgement.
Progress is saved after every query. Unjudged jobs count as non-relevant in the eval,
the standard pooling assumption, so recall here is "recall within the judged pool".
"""

import argparse
import random
import textwrap

from pydantic import BaseModel, Field

from app.db import make_pool, run
from app.evals.dataset import HANDLABELED_PATH, job_key, load, save
from app.llm import generate_structured
from app.search.models import SearchMode, SearchRequest
from app.search.service import search

DRAFT_QUERIES = [
    "remote senior python backend engineer",
    "entry level marketing coordinator",
    "registered nurse ICU night shift",
    "data analyst SQL tableau",
    "machine learning engineer working on LLMs",
    "part time customer service work from home",
    "staff accountant CPA",
    "react frontend developer",
    "devops engineer kubernetes aws",
    "sales development representative for a SaaS startup",
    "UX designer figma",
    "construction project manager",
    "warehouse associate forklift",
    "rust systems programming",
    "HR generalist",
    "financial analyst FP&A",
    "electrical engineer power systems",
    "high school math teacher",
    "physical therapist outpatient clinic",
    "I want to move from teaching into instructional design",
    "junior software developer first job after bootcamp",
    "product manager B2B fintech",
    "truck driver CDL class A",
    "legal assistant at a law firm",
    "data engineer airflow spark",
    "social media manager",
    "medical billing and coding remote",
    "mobile developer iOS swift",
    "executive assistant to the CEO",
    "AI research scientist computer vision",
]

MODES = [SearchMode.keyword, SearchMode.vector, SearchMode.hybrid, SearchMode.hybrid_rerank]


def _ensure_drafts():
    from app.evals.dataset import EvalQuery

    qs = load(HANDLABELED_PATH)
    if not qs:
        qs = [EvalQuery(qid=f"hl-{i:03d}", query=q, set="handlabeled") for i, q in enumerate(DRAFT_QUERIES, 1)]
        save(HANDLABELED_PATH, qs)
        print(f"created {HANDLABELED_PATH} with {len(qs)} draft queries (edit freely)")
    return qs


async def _pool_results(pool, q) -> list:
    seen, hits = set(), []
    for mode in MODES:
        resp = await search(pool, SearchRequest(query=q.query, mode=mode, k=10, filters=q.filters))
        for h in resp.results:
            if h.job_id not in seen:
                seen.add(h.job_id)
                hits.append(h)
    return hits


async def _keys(pool, ids: list[int]) -> dict[int, str]:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT id, source, external_id FROM jobs WHERE id = ANY(%s)", (ids,))
        return {r["id"]: job_key(r["source"], r["external_id"]) for r in await cur.fetchall()}


def _ask(prompt: str, default: int | None) -> str:
    while True:
        a = input(prompt).strip().lower()
        if a == "" and default is not None:
            return str(default)
        if a in {"0", "1", "2", "s", "q"}:
            return a


class _Grade(BaseModel):
    n: int
    grade: int = Field(ge=0, le=2)


class _Grades(BaseModel):
    grades: list[_Grade]


async def _suggest(query: str, hits: list) -> dict[int, int]:
    """LLM pre-grades the pool. The human confirms or overrides every one, so this
    only saves keystrokes; the final labels are still human judgements."""
    docs = "\n\n".join(
        f"[{n}] {h.title} | {h.company} | {h.location or ''}{' (remote)' if h.is_remote else ''}\n{h.snippet}"
        for n, h in enumerate(hits, 1)
    )
    prompt = (
        f"Job search query: {query!r}\n\nGrade each job: 2 = great match for this searcher, "
        f"1 = partially relevant / acceptable, 0 = not relevant.\n\n{docs}"
    )
    try:
        resp = await generate_structured(prompt, _Grades, temperature=0.0)
        return {g.n: g.grade for g in resp.grades}
    except Exception as e:  # labelling still works without suggestions
        print(f"(no LLM suggestions: {e})")
        return {}


async def label(redo: str | None, assist: bool = True) -> None:
    qs = _ensure_drafts()
    pool = make_pool()
    await pool.open()
    try:
        for q in qs:
            if q.qrels and q.qid != redo:
                continue
            if redo and q.qid != redo:
                continue
            hits = await _pool_results(pool, q)
            random.shuffle(hits)
            keys = await _keys(pool, [h.job_id for h in hits])
            suggestions = await _suggest(q.query, hits) if assist else {}
            print("\n" + "=" * 80 + f"\n[{q.qid}] {q.query}   ({len(hits)} pooled results)\n" + "=" * 80)
            judged: dict[str, int] = {}
            for n, h in enumerate(hits, 1):
                loc = f"{h.location or ''}{' · remote' if h.is_remote else ''}"
                print(f"\n({n}/{len(hits)}) {h.title}  |  {h.company}  |  {loc}  [{h.seniority or '-'}]")
                print(textwrap.indent(textwrap.fill(h.snippet, 100), "    "))
                sug = suggestions.get(n)
                hint = f" [Enter = {sug}]" if sug is not None else ""
                a = _ask(f"    relevance 2/1/0{hint}, s=skip query, q=quit > ", sug)
                if a == "q":
                    print("saved, bye")
                    return
                if a == "s":
                    judged = {}
                    break
                judged[keys[h.job_id]] = int(a)
            if judged:
                q.qrels = judged
                save(HANDLABELED_PATH, qs)
                print(f"saved {q.qid}: {sum(v > 0 for v in judged.values())} relevant of {len(judged)}")
    finally:
        await pool.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--redo", help="re-label one qid")
    p.add_argument("--no-assist", action="store_true", help="don't pre-grade with the LLM")
    a = p.parse_args()
    run(label(a.redo, assist=not a.no_assist))


if __name__ == "__main__":
    main()
