from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class JobPosting(BaseModel):
    """Source-agnostic job record. Every source adapter must produce these."""

    source: str
    external_id: str
    url: str | None = None
    title: str = Field(min_length=2)
    company: str | None = None
    location: str | None = None
    is_remote: bool = False
    seniority: str | None = None
    employment_type: str | None = None
    category: str | None = None
    skills: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    description: str = Field(min_length=20)
    posted_at: datetime | None = None

    @field_validator("title", "company", "location", "description", mode="before")
    @classmethod
    def strip(cls, v):
        return v.strip() if isinstance(v, str) else v
