"""Gateway client: every LLM call in the app is built here.

Agents never instantiate provider SDKs directly. They ask this module for a
chat model by *tier* (``fast`` or ``reasoning``); the model is an
OpenAI-compatible client pointed at the LiteLLM proxy, which owns the actual
cost/latency routing, fallbacks, and budgets (see ``litellm.config.yaml``).

This keeps provider choice out of the agent code: swapping models or providers
is a gateway-config change, not a code change.
"""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

ModelTier = Literal["fast", "reasoning"]


def _resolve_model(tier: ModelTier, settings: Settings) -> str:
    """Map a logical tier to a LiteLLM model group, preferring Anthropic when available.

    We prefer Anthropic (Claude) for reasoning because it is measurably stronger at
    multi-step tool-use and root-cause analysis (verified in our evals). But we only
    route there when ``ANTHROPIC_API_KEY`` is configured; otherwise we use the
    OpenAI-only group so the whole system still runs on an OpenAI key alone — with no
    failed Anthropic attempts and no wasted fallback latency.

    FUTURE: when an Anthropic key is added, the reasoning tier automatically upgrades
    to Claude Sonnet with zero code change (this function already handles it).
    """
    has_anthropic = bool(settings.anthropic_api_key)
    if tier == "reasoning":
        return settings.model_reasoning if has_anthropic else settings.model_reasoning_openai
    return settings.model_fast if has_anthropic else settings.model_fast_openai


def get_chat_model(
    tier: ModelTier = "fast",
    *,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 2,
    settings: Settings | None = None,
    **kwargs: object,
) -> ChatOpenAI:
    """Build a chat model for the given tier, routed through the LiteLLM gateway.

    Args:
        tier: ``"fast"`` for triage/classification, ``"reasoning"`` for RCA.
        temperature: sampling temperature (default deterministic).
        timeout: per-request timeout in seconds.
        max_retries: client-side retries (the gateway also has its own).
        settings: optional override, mainly for tests.
        **kwargs: extra params forwarded to :class:`ChatOpenAI`.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        model=_resolve_model(tier, settings),
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )
