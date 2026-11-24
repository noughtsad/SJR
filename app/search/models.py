from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    keyword = "keyword"
    vector = "vector"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


class SearchFilters(BaseModel):
    remote: bool | None = None
    location: str | None = Field(None, description="case-insensitive substring match")
    seniority: list[str] | None = None
    skills: list[str] | None = Field(None, description="job must have at least one")
    posted_within_days: int | None = Field(None, ge=1)
    sources: list[str] | None = None
    exclude_job_ids: list[int] | None = Field(None, description="e.g. jobs the user already saved or dismissed")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: SearchMode = SearchMode.hybrid_rerank
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(10, ge=1, le=50)
    model: str | None = Field(None, description="embedding model key; defaults to settings")


class Candidate(BaseModel):
    """A job id carrying the score/rank it got at every stage it passed through,
    so a response shows *why* a result landed where it did."""

    job_id: int
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class JobHit(Candidate):
    title: str
    company: str | None
    location: str | None
    is_remote: bool
    seniority: str | None
    skills: list[str]
    url: str | None
    posted_at: datetime | None
    snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    model: str | None
    results: list[JobHit]
    timings_ms: dict[str, float]
