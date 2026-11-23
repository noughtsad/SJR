from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.ingestion.models import JobPosting


class JobSource(ABC):
    """A source adapter turns raw records into JobPostings.

    full_snapshot: every fetch returns *all* currently open jobs, so anything
        not seen in a successful run has been delisted and can be expired.
    ttl_days: for incremental / static sources, expire by posting age instead
        (None disables TTL expiry, e.g. for a historical Kaggle dump).
    """

    name: str
    full_snapshot: bool = False
    ttl_days: int | None = None

    @abstractmethod
    def fetch_raw(self) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    def to_posting(self, raw: dict[str, Any]) -> JobPosting: ...
