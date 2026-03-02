"""
LLM Router
==========
Swappable LLM factory supporting Anthropic, OpenAI, Groq, and xAI.
Implements automatic fallback with retry logic.
"""

import structlog
from langchain_core.language_models import BaseChatModel

from config.settings import LLMProvider, Settings, get_settings

logger = structlog.get_logger(__name__)


def build_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """
    Factory function for building a LangChain-compatible LLM.

    Supports: Anthropic Claude, OpenAI GPT, Groq (Llama), xAI Grok.
    Falls back to FALLBACK_LLM_PROVIDER if primary fails.

    Args:
        provider: Override LLM_PROVIDER from settings
        model: Override LLM_MODEL from settings
        temperature: Override temperature
        max_tokens: Override max_tokens
        settings: Optional pre-loaded settings

    Returns:
        Configured BaseChatModel instance
    """
    cfg = settings or get_settings()
    _provider = provider or cfg.llm_provider
    _model = model or cfg.llm_model
    _temperature = temperature if temperature is not None else cfg.llm_temperature
    _max_tokens = max_tokens or cfg.llm_max_tokens

    logger.info(
        "building_llm",
        provider=_provider,
        model=_model,
        temperature=_temperature,
    )

    try:
        return _build_llm_for_provider(_provider, _model, _temperature, _max_tokens, cfg)
    except Exception as e:
        logger.warning(
            "primary_llm_failed_using_fallback",
            error=str(e),
            fallback_provider=cfg.fallback_llm_provider,
        )
        return _build_llm_for_provider(
            cfg.fallback_llm_provider,
            cfg.fallback_llm_model,
            _temperature,
            _max_tokens,
            cfg,
        )


def _build_llm_for_provider(
    provider: LLMProvider,
    model: str,
    temperature: float,
    max_tokens: int,
    cfg: Settings,
) -> BaseChatModel:
    """Internal factory — creates LLM for a specific provider."""

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        if not cfg.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=cfg.anthropic_api_key,
        )  # type: ignore[call-arg]

    elif provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=cfg.openai_api_key,
        )

    elif provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq

        if not cfg.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=cfg.groq_api_key,
        )

    elif provider == LLMProvider.XAI:
        # xAI Grok via OpenAI-compatible API
        from langchain_openai import ChatOpenAI

        if not cfg.xai_api_key:
            raise ValueError("XAI_API_KEY not set")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=cfg.xai_api_key,
            base_url="https://api.x.ai/v1",
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def build_embedding_model(settings: Settings | None = None):
    """Build embedding model for vector store operations."""
    cfg = settings or get_settings()

    if cfg.embedding_provider.value == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=cfg.openai_api_key,
        )

    elif cfg.embedding_provider.value == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif cfg.embedding_provider.value == "nomic":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model="nomic-embed-text")

    raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")
