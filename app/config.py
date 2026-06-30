"""Central application settings, loaded from environment / .env.

All configuration flows through a single ``Settings`` object so the rest of the
codebase never reads ``os.environ`` directly. Import the cached accessor:

    from app.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed view of the project's environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"

    # --- LLM provider keys (consumed by the gateway) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- LiteLLM gateway ---
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "sk-local-dev"
    # Logical model tiers -> resolve to model groups in litellm.config.yaml.
    model_fast: str = "fast"
    model_reasoning: str = "reasoning"

    # --- RAG / Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "sre_runbooks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Observability: LangSmith ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "sre-incident-copilot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Observability: Langfuse ---
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_ready(self) -> bool:
        """True only when Langfuse is enabled *and* fully configured."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    @property
    def langsmith_ready(self) -> bool:
        """True only when LangSmith tracing is enabled *and* has a key."""
        return bool(self.langsmith_tracing and self.langsmith_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
