"""
Unit tests for config/settings.py
"""

from __future__ import annotations

import pytest


class TestSettingsDefaults:
    def test_defaults_load_without_env(self, monkeypatch):
        # Clear any env vars that might interfere with defaults
        monkeypatch.delenv("API_SECRET_KEY", raising=False)
        monkeypatch.delenv("API_DEBUG", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        # Settings requires api_debug=True OR a non-default secret in production
        # Force debug mode so the production validator does not fire
        monkeypatch.setenv("API_DEBUG", "true")

        from config.settings import Settings
        s = Settings()
        assert s.llm_provider.value in ("anthropic", "openai", "groq")
        assert s.api_port == 8000
        assert s.log_level.value == "INFO"
        assert s.retrieval_k == 10

    def test_get_allowed_origins_splits_commas(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://a.com,http://b.com, http://c.com")

        from config.settings import Settings
        s = Settings()
        origins = s.get_allowed_origins()
        assert origins == ["http://a.com", "http://b.com", "http://c.com"]

    def test_get_allowed_origins_handles_single_value(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8501")

        from config.settings import Settings
        s = Settings()
        assert s.get_allowed_origins() == ["http://localhost:8501"]

    def test_get_active_llm_api_key_returns_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        from config.settings import Settings
        s = Settings()
        assert s.get_active_llm_api_key() == "sk-ant-test"

    def test_get_active_llm_api_key_returns_openai_key(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        from config.settings import Settings
        s = Settings()
        assert s.get_active_llm_api_key() == "sk-openai-test"

    def test_is_production_true_when_debug_false(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "false")
        monkeypatch.setenv("API_SECRET_KEY", "a-secure-production-key-abc123")

        from config.settings import Settings
        s = Settings()
        assert s.is_production() is True

    def test_is_production_false_when_debug_true(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")

        from config.settings import Settings
        s = Settings()
        assert s.is_production() is False


class TestSettingsProductionValidator:
    def test_raises_when_secret_key_is_default_in_production(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "false")
        monkeypatch.setenv("API_SECRET_KEY", "change-me")

        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="api_secret_key must be changed"):
            Settings()

    def test_raises_when_secret_key_is_example_placeholder(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "false")
        monkeypatch.setenv("API_SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")

        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="api_secret_key must be changed"):
            Settings()

    def test_passes_with_custom_secret_key_in_production(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "false")
        monkeypatch.setenv("API_SECRET_KEY", "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")

        from config.settings import Settings
        s = Settings()
        assert s.api_secret_key == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"

    def test_allows_default_key_in_debug_mode(self, monkeypatch):
        monkeypatch.setenv("API_DEBUG", "true")
        monkeypatch.setenv("API_SECRET_KEY", "change-me")

        from config.settings import Settings
        s = Settings()
        assert s.api_secret_key == "change-me"
