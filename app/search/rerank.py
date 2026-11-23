"""Stage 4: cross-encoder reranking of the fused top-N.

Bi-encoders (stage 2) embed query and document separately, which is fast but
lossy. A cross-encoder reads the (query, document) pair together through full
attention, which is much more accurate and far too slow to run over the whole
index. So it only rescores the ~30 candidates that fusion already put on top.
"""

import asyncio
from functools import cache

from app.config import get_settings
from app.search.models import Candidate


@cache
def _model():
    from sentence_transformers import CrossEncoder

    from app.device import get_device, model_kwargs

    s = get_settings()
    return CrossEncoder(
        s.reranker_model, device=get_device(), max_length=s.rerank_max_tokens, model_kwargs=model_kwargs()
    )


def _score(query: str, docs: list[str]) -> list[float]:
    return [float(s) for s in _model().predict([(query, d) for d in docs], batch_size=16, show_progress_bar=False)]


async def rerank(query: str, candidates: list[Candidate], docs: dict[int, str]) -> list[Candidate]:
    if not candidates:
        return []
    scores = await asyncio.to_thread(_score, query, [docs[c.job_id] for c in candidates])
    out = []
    for c, s in zip(candidates, scores, strict=True):
        c = c.model_copy()
        c.rerank_score = s
        out.append(c)
    return sorted(out, key=lambda c: -c.rerank_score)


def warmup() -> None:
    _model()
