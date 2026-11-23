"""HN "Ask HN: Who is hiring?" monthly threads via the free Algolia API.

Each top-level comment is one posting, by convention headed
    Company | Role | Location | REMOTE/ONSITE | Salary ...
The header is free text, so parsing is heuristic and some fields may stay empty.

Snapshot semantics: the last `months` threads together are the current open
set. When a new month starts, the oldest thread drops out of the window and
its posts expire on the next run.
"""

import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.ingestion.models import JobPosting
from app.ingestion.normalize import extract_skills, html_to_text, infer_seniority, looks_remote
from app.ingestion.sources.base import JobSource
from app.ingestion.sources.remotive import parse_salary

ALGOLIA = "https://hn.algolia.com/api/v1"

_ROLE = re.compile(
    r"\b(engineer|engineers|developer|developers|scientist|designer|manager|lead|architect|analyst|"
    r"researcher|intern|sre|devops|admin|administrator|consultant|specialist|head of|director|cto|"
    r"founding|recruiter|marketer|writer|technician|ml|ai|frontend|backend|full[- ]?stack|phd|roles?|positions?)\b",
    re.I,
)
_WORK_MODE = re.compile(r"^\s*(remote|onsite|on-site|hybrid|in[- ]office)\b", re.I)
_EMPLOYMENT = re.compile(r"\b(full[- ]?time|part[- ]?time|contract|contractor|freelance|internship)\b", re.I)
_MONEY = re.compile(r"[$€£]|\b\d+\s?k\b|\b(salary|equity|usd|eur|gbp)\b", re.I)
_URLISH = re.compile(r"https?://|www\.|\.com\b|\.io\b", re.I)


def parse_header(header: str) -> dict:
    segs = [s.strip() for s in header.split("|") if s.strip()]
    out = {"company": segs[0] if segs else None, "role": None, "location": None,
           "employment_type": None, "salary": None}  # fmt: skip
    locs = []
    for seg in segs[1:]:
        if _URLISH.search(seg):
            continue
        if out["salary"] is None and _MONEY.search(seg) and re.search(r"\d", seg):
            out["salary"] = seg
        elif _EMPLOYMENT.search(seg) and len(seg) < 40:
            out["employment_type"] = out["employment_type"] or _EMPLOYMENT.search(seg).group(1).lower()
        elif out["role"] is None and _ROLE.search(seg):
            out["role"] = seg
        elif _WORK_MODE.match(seg):
            rest = _WORK_MODE.sub("", seg).strip(" ()-,:")
            if rest:
                locs.append(rest)
        elif len(seg) < 60:
            locs.append(seg)
    out["location"] = "; ".join(locs) or None
    return out


class HNHiringSource(JobSource):
    name = "hn_hiring"
    full_snapshot = True
    user_generated = True

    def __init__(self, months: int = 2):
        self.months = months

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
    async def _get(self, client: httpx.AsyncClient, url: str, **params) -> dict:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def fetch_raw(self) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=get_settings().http_timeout_s) as client:
            found = await self._get(
                client, f"{ALGOLIA}/search_by_date", tags="story,author_whoishiring", hitsPerPage=20
            )
            threads = [h for h in found["hits"] if h["title"].startswith("Ask HN: Who is hiring?")][: self.months]
            if not threads:
                raise RuntimeError("no 'Who is hiring' threads found")
            for t in threads:
                item = await self._get(client, f"{ALGOLIA}/items/{t['objectID']}")
                for c in item.get("children", []):
                    if c.get("text"):  # deleted / flagged comments have no text
                        yield {**c, "thread_title": t["title"]}

    def to_posting(self, raw: dict[str, Any]) -> JobPosting:
        html = raw["text"]
        header_html, _, _ = html.partition("<p>")
        header = html_to_text(header_html).replace("\n", " ")
        h = parse_header(header)
        company = h["company"][:120] if h["company"] else None
        title = (h["role"] or f"Role at {company or 'unknown company'}")[:200]
        description = html_to_text(html.replace("<p>", "\n<p>"))
        smin, smax, cur = parse_salary(h["salary"])
        return JobPosting(
            source=self.name,
            external_id=str(raw["id"]),
            url=f"https://news.ycombinator.com/item?id={raw['id']}",
            title=title,
            company=company,
            location=h["location"],
            is_remote=looks_remote(header),
            seniority=infer_seniority(title),
            employment_type=h["employment_type"],
            skills=extract_skills(title, description),
            salary_min=smin,
            salary_max=smax,
            salary_currency=cur,
            description=description,
            posted_at=datetime.fromtimestamp(raw["created_at_i"], tz=UTC),
        )
