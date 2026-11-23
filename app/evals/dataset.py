"""Eval set I/O. Jobs are referenced by "source:external_id", which is stable
across database rebuilds, unlike the serial jobs.id."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.search.models import SearchFilters

EVAL_DIR = Path(__file__).resolve().parents[2] / "evals" / "retrieval"
SYNTHETIC_PATH = EVAL_DIR / "synthetic.jsonl"
HANDLABELED_PATH = EVAL_DIR / "handlabeled.jsonl"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"


class EvalQuery(BaseModel):
    qid: str
    query: str
    set: str  # "synthetic" | "handlabeled"
    style: str | None = None  # synthetic: keyword | natural | skills
    filters: SearchFilters = Field(default_factory=SearchFilters)
    qrels: dict[str, int] = Field(default_factory=dict)  # "source:external_id" -> 0/1/2


def job_key(source: str, external_id: str) -> str:
    return f"{source}:{external_id}"


def load(path: Path) -> list[EvalQuery]:
    if not path.exists():
        return []
    return [EvalQuery.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save(path: Path, queries: list[EvalQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(q.model_dump(mode="json", exclude_defaults=False)) for q in queries) + "\n",
        encoding="utf-8",
    )


async def key_to_id_map(pool, keys: set[str]) -> dict[str, int]:
    """Resolve source:external_id -> active jobs.id (expired jobs drop out)."""
    pairs = [k.split(":", 1) for k in keys]
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT j.id, j.source, j.external_id
               FROM jobs j JOIN unnest(%s::text[], %s::text[]) AS k(source, external_id)
                 ON j.source = k.source AND j.external_id = k.external_id
               WHERE j.is_active""",
            ([p[0] for p in pairs], [p[1] for p in pairs]),
        )
        return {job_key(r["source"], r["external_id"]): r["id"] for r in await cur.fetchall()}
