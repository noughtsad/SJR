"""Kaggle "LinkedIn Job Postings (2023-2024)" dataset (arshkon/linkedin-job-postings),
file postings.csv. A static historical dump: no snapshot expiry, no TTL.

Download: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
Other CSVs work too; pass a different column mapping to KaggleCsvSource.
"""

import csv
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingestion.models import JobPosting
from app.ingestion.normalize import clean_text, extract_skills, infer_seniority, looks_remote
from app.ingestion.sources.base import JobSource

DEFAULT_COLUMNS = {
    "id": "job_id",
    "title": "title",
    "company": "company_name",
    "description": "description",
    "location": "location",
    "url": "job_posting_url",
    "remote": "remote_allowed",
    "experience": "formatted_experience_level",
    "work_type": "formatted_work_type",
    "skills": "skills_desc",
    "min_salary": "min_salary",
    "max_salary": "max_salary",
    "currency": "currency",
    "listed_time": "listed_time",  # epoch millis
}


def _float(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


class KaggleCsvSource(JobSource):
    name = "kaggle_linkedin"
    full_snapshot = False
    ttl_days = None

    def __init__(self, path: Path, limit: int | None = None, columns: dict[str, str] | None = None):
        self.path = path
        self.limit = limit
        self.cols = {**DEFAULT_COLUMNS, **(columns or {})}

    async def fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with self.path.open(encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if self.limit is not None and i >= self.limit:
                    break
                yield row

    def to_posting(self, raw: dict[str, Any]) -> JobPosting:
        c = self.cols
        g = lambda key: (raw.get(c[key]) or "").strip() or None  # noqa: E731
        title = g("title") or ""
        description = clean_text(g("description") or "")
        location = g("location")
        listed = _float(g("listed_time"))
        remote_flag = g("remote")
        return JobPosting(
            source=self.name,
            external_id=g("id") or "",
            url=g("url"),
            title=title,
            company=g("company"),
            location=location,
            is_remote=remote_flag in ("1", "1.0", "True", "true") or looks_remote(title, location),
            seniority=infer_seniority(title, hint=g("experience")),
            employment_type=g("work_type"),
            skills=extract_skills(title, description, g("skills")),
            salary_min=_float(g("min_salary")),
            salary_max=_float(g("max_salary")),
            salary_currency=g("currency"),
            description=description,
            posted_at=datetime.fromtimestamp(listed / 1000, tz=UTC) if listed else None,
        )
