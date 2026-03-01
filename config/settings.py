"""
BiasGuard Configuration
=======================
All settings are environment-driven via .env.
Uses Pydantic Settings for type-safe config with validation.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    XAI = "xai"


class VectorDB(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    NOMIC = "nomic"
    HUGGINGFACE = "huggingface"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=32768)

    fallback_llm_provider: LLMProvider = LLMProvider.GROQ
    fallback_llm_model: str = "llama-3.1-70b-versatile"

    # ── API Keys ───────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    xai_api_key: str = ""

    # ── Embeddings ─────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Vector DB ──────────────────────────────────────────
    vector_db: VectorDB = VectorDB.CHROMA
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "bias_patterns"

    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "biasguard"

    # ── API ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me"
    api_debug: bool = False
    allowed_origins: list[str] = ["http://localhost:8501"]
    rate_limit_per_minute: int = 60

    # ── Observability ──────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "biasguard-dev"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_enabled: bool = False

    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # ── Logging ────────────────────────────────────────────
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Path = Path("./logs/biasguard.log")

    # ── Reports ────────────────────────────────────────────
    reports_dir: Path = Path("./data/outputs")
    pdf_generation_enabled: bool = True

    # ── Features ───────────────────────────────────────────
    enable_async_processing: bool = False
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # ── RAG ────────────────────────────────────────────────
    retrieval_k: int = 10
    retrieval_score_threshold: float = 0.7

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("chroma_persist_dir", "log_file", "reports_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    def get_active_llm_api_key(self) -> str:
        key_map = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.XAI: self.xai_api_key,
        }
        return key_map[self.llm_provider]

    def is_production(self) -> bool:
        return not self.api_debug


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
