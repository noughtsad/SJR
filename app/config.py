from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://jobs:jobs@localhost:5433/jobs"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # Embeddings: key from app.embeddings.registry.MODELS
    embedding_model: str = "bge-small"
    embedding_batch_size: int = 64
    # "auto" = CUDA if available, else CPU. Force with DEVICE=cpu / DEVICE=cuda.
    device: str = "auto"

    # Reranker (cross-encoder). MiniLM-L6 (22M params) is CPU-friendly;
    # BAAI/bge-reranker-base (278M) is usually more accurate but much slower.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_max_tokens: int = 384
    rerank_top_n: int = 30

    # Search
    keyword_max_df: float = 0.15  # drop query terms appearing in more than this share of jobs
    rrf_k: int = 60
    rrf_w_keyword: float = 1.0
    rrf_w_vector: float = 1.0
    candidate_pool: int = 50  # how many results each retriever contributes to fusion
    hnsw_ef_search: int = 100

    # Gemini (hosted embeddings now; LLM features in later layers)
    gemini_api_key: str | None = None

    # Ingestion
    remotive_url: str = "https://remotive.com/api/remote-jobs"
    http_timeout_s: float = 30.0
    expire_after_days: int = 45  # TTL for sources that are not full snapshots
    # A snapshot source that suddenly returns far fewer jobs than are active is more
    # likely a broken API response than a mass delisting. Below this ratio we skip expiry.
    min_snapshot_ratio: float = 0.5

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # The project's .env beats machine-wide environment variables (a stale global
        # GEMINI_API_KEY shouldn't override this project's key). Docker/CI have no .env,
        # so they still configure everything through environment variables.
        return init_settings, dotenv_settings, env_settings, file_secret_settings


@lru_cache
def get_settings() -> Settings:
    return Settings()
