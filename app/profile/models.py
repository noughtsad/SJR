from typing import Literal

from pydantic import BaseModel, Field, field_validator

Seniority = Literal["intern", "junior", "mid", "senior", "lead", "executive"]
RemotePreference = Literal["remote_only", "hybrid_or_remote", "any"]


class CandidateProfile(BaseModel):
    """What the LLM extracts from a resume. Every field must be stated in, or
    directly computable from, the resume; nothing is guessed."""

    headline: str = Field(description="One line, e.g. 'Backend engineer, 6 years, Python and Go'")
    target_roles: list[str] = Field(
        min_length=1, max_length=5,
        description="Job titles this person is a fit for, most likely first, based on their recent roles",
    )  # fmt: skip
    seniority: Seniority
    years_experience: float = Field(ge=0, le=60, description="Total professional years, from the dates listed")
    skills: list[str] = Field(
        max_length=40, description="Technical and professional skills explicitly mentioned in the resume"
    )
    domains: list[str] = Field(default_factory=list, max_length=10, description="Industries / problem areas")
    locations: list[str] = Field(default_factory=list, max_length=5, description="Where they live or want to work")
    remote_preference: RemotePreference = Field(
        "any", description="Only if the resume states a preference; otherwise 'any'"
    )
    education: list[str] = Field(default_factory=list, max_length=5)
    summary: str = Field(max_length=800, description="2-3 sentences: what they have built and are strongest at")

    @field_validator("skills", "domains", "target_roles")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for s in (x.strip() for x in v):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out


class StoredProfile(BaseModel):
    user_id: str
    profile: CandidateProfile
    canonical_skills: list[str]  # profile skills mapped onto the job-side skill dictionary
    version: int
