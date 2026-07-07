"""Observability wiring (Layer 7) — LangSmith + Langfuse.

Two backends, each with a distinct role and both enabled by configuration only:

* **LangSmith** — auto-traces LangChain/LangGraph runs when the ``LANGSMITH_*``
  env vars are set. We simply export them from settings; no callback needed.
* **Langfuse** — attached via a LangChain ``CallbackHandler`` passed in the run
  config. We initialize the client from settings, then hand back the handler.

Everything is gated by the ``*_ready`` flags on ``Settings`` so the system runs
fine (and tests stay offline) with no keys configured.

Typical use (Layer 8)::

    setup_observability(settings)
    config = runnable_config(thread_id="INC-1001", settings=settings)
    graph.invoke(state, config=config)
"""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings, get_settings


def configure_langsmith(settings: Settings | None = None) -> bool:
    """Export LANGSMITH_* env vars so LangChain auto-traces. Returns enabled?."""
    settings = settings or get_settings()
    if not settings.langsmith_ready:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    return True


def get_langfuse_handler(settings: Settings | None = None) -> Any | None:
    """Return a Langfuse LangChain callback handler, or None if not configured."""
    settings = settings or get_settings()
    if not settings.langfuse_ready:
        return None
    # Imported lazily so the dependency is only touched when actually enabled.
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler()


def get_callbacks(settings: Settings | None = None) -> list[Any]:
    """Callbacks to attach to a run (Langfuse handler if enabled)."""
    handler = get_langfuse_handler(settings)
    return [handler] if handler is not None else []


def setup_observability(settings: Settings | None = None) -> list[Any]:
    """Configure LangSmith env + return run callbacks (Langfuse). Call once at start."""
    settings = settings or get_settings()
    configure_langsmith(settings)
    return get_callbacks(settings)


def runnable_config(
    thread_id: str,
    *,
    settings: Settings | None = None,
    extra_callbacks: list[Any] | None = None,
    recursion_limit: int = 50,
) -> dict[str, Any]:
    """Build a LangGraph run config: thread id (checkpointer) + trace callbacks."""
    callbacks = get_callbacks(settings)
    if extra_callbacks:
        callbacks = [*callbacks, *extra_callbacks]
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config
