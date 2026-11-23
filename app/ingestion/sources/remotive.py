"""Remotive public API (no key). It returns every open remote job in one
response, so it is treated as a full snapshot.

Remotive asks clients to call it at most a few times per day, so each raw
response is saved to data/raw/ and can be replayed with --from-file.
"""

import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.ingestion.models import JobPosting
from app.ingestion.normalize import extract_skills, html_to_text, infer_seniority
from app.ingestion.sources.base import JobSource

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

_MONEY = re.compile(r"(\d[\d,.]*)\s*(k)?", re.I)


def parse_salary(s: str | None) -> tuple[float | None, float | None, str | None]:
    """Remotive salary is free text, e.g. '$120k - $150k', '50,000 - 70,000 EUR'."""
    if not s:
        return None, None, None
    nums = []
    for amount, k in _MONEY.findall(s):
        try:
            v = float(amount.replace(",", ""))
        except ValueError:
            continue
        if k:
            v *= 1000
        if v >= 1000:  # ignore stray small numbers
            nums.append(v)
    currency = "USD" if "$" in s else "EUR" if ("€" in s or "EUR" in s.upper()) else "GBP" if "£" in s else None
    if not nums:
        return None, None, currency
    return min(nums), max(nums), currency


class RemotiveSource(JobSource):
    name = "remotive"
    full_snapshot = True

    def __init__(self, from_file: Path | None = None, save_raw: bool = True):
        self.from_file = from_file
        self.save_raw = save_raw

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
    async def _download(self) -> dict:
        s = get_settings()
        async with httpx.AsyncClient(timeout=s.http_timeout_s) as client:
            r = await client.get(s.remotive_url, headers={"User-Agent": "semantic-job-search/0.1"})
            r.raise_for_status()
            return r.json()

    async def fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        if self.from_file:
            payload = json.loads(self.from_file.read_text(encoding="utf-8"))
        else:
            payload = await self._download()
            if self.save_raw:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                (RAW_DIR / f"remotive_{stamp}.json").write_text(json.dumps(payload), encoding="utf-8")
        for job in payload.get("jobs", []):
            yield job

    def to_posting(self, raw: dict[str, Any]) -> JobPosting:
        title = raw.get("title") or ""
        description = html_to_text(raw.get("description"))
        tags = raw.get("tags") or []
        smin, smax, cur = parse_salary(raw.get("salary"))
        posted = raw.get("publication_date")
        return JobPosting(
            source=self.name,
            external_id=str(raw["id"]),
            url=raw.get("url"),
            title=title,
            company=raw.get("company_name"),
            # candidate_required_location is where the candidate may live, e.g. "USA Only", "Worldwide"
            location=raw.get("candidate_required_location") or "Worldwide",
            is_remote=True,
            seniority=infer_seniority(title),
            employment_type=raw.get("job_type") or None,
            category=raw.get("category"),
            skills=extract_skills(title, description, extra=tags),
            salary_min=smin,
            salary_max=smax,
            salary_currency=cur,
            description=description,
            posted_at=datetime.fromisoformat(posted).replace(tzinfo=UTC) if posted else None,
        )
