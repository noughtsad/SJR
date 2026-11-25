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

    # Gemini (hosted embeddings, profile extraction, explanations)
    gemini_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_rpm: int | None = None  # shared requests/minute cap across all LLM calls (free tier: set ~14)

    # Recommendations. Rocchio: q = a*profile + b*mean(saved) - c*mean(dismissed).
    # 1.0 / 0.75 / 0.15 are the textbook defaults (Manning et al., IIR ch. 9).
    rocchio_alpha: float = 1.0
    rocchio_beta: float = 0.75
    rocchio_gamma: float = 0.15
    rec_candidates: int = 100  # vector neighbours re-scored per request
    rec_skill_weight: float = 0.10  # bonus for share of the job's skills the user has
    rec_seniority_penalty: float = 0.03  # per level of mismatch
    rec_max_per_company: int = 2

    # Career agent (Layer 4). Defaults to llm_model; a separate knob because the agent makes
    # several calls per turn and free-tier quotas are per model.
    agent_model: str | None = "gemini-3.5-flash-lite"
    agent_max_messages: int = 40  # most recent messages of a thread the model sees
    agent_recursion_limit: int = 25  # graph steps per turn (each tool round trip is 2)
    agent_max_tool_rounds: int = 4  # then the model must answer without tools

    # Evals (Layer 5). Prices are USD per 1M tokens for the cost estimate; set them to the
    # current list price of AGENT_MODEL (defaults are placeholders in the flash-lite range).
    judge_model: str = "gemini-3.5-flash-lite"
    price_input_per_mtok: float = 0.10
    price_output_per_mtok: float = 0.40

    # Ingestion
    remotive_url: str = "https://remotive.com/api/remote-jobs"
    http_timeout_s: float = 30.0
    expire_after_days: int = 45  # TTL for sources that are not full snapshots
    # A snapshot source that suddenly returns far fewer jobs than are active is more
    # likely a broken API response than a mass delisting. Below this ratio we skip expiry.
    min_snapshot_ratio: float = 0.5

    # API hardening (Layer 6)
    log_format: str = "text"  # "json" in containers: one JSON object per line, with request_id
    log_level: str = "INFO"
    rate_limit_per_min: int = 120  # per client IP; 0 disables
    rate_limit_llm_per_min: int = 10  # /chat, resume upload, explain: they spend LLM quota
    trust_proxy: bool = False  # take the client IP from X-Forwarded-For (only behind your own proxy)
    request_timeout_s: float = 15.0
    db_statement_timeout_ms: int = 10_000  # API pool only; CLI jobs (ingest, indexing) are unbounded
    search_cache_ttl_s: float = 60.0
    search_cache_size: int = 512
    cors_origins: list[str] = ["http://localhost:3000"]

    # Langfuse tracing: enabled only when both keys are set
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"

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
