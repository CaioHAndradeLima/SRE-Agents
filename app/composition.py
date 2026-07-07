"""Composition root — wire every layer into a runnable copilot.

This is the one place that knows how the layers fit together (the dependency-
injection container). Everything above stays decoupled; here we choose concrete
implementations based on settings:

    settings → SRE environment (mock/real) + RAG retriever
             → tools → permission guard → agents → supervisor graph

Swapping the mock for real HTTP integrations, or the offline embedder for
fastembed, happens here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient

from app.agents.factory import build_all_agents
from app.agents.registry import AgentRegistry, load_agent_registry
from app.config import Settings, get_settings
from app.graph.supervisor import build_supervisor_graph
from app.integrations.mock import MockSRE
from app.integrations.protocols import SREEnvironment
from app.rag.embeddings import Embedder, FastEmbedEmbedder
from app.rag.retriever import RunbookRetriever
from app.security.guard import ApprovalContext, PermissionGuard
from app.tools.catalog import build_tools, tool_harm_tier

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNBOOKS = ROOT / "data" / "runbooks"


@dataclass
class IncidentCopilot:
    """Everything needed to run and inspect one incident session."""

    graph: CompiledStateGraph
    guard: PermissionGuard
    approval_ctx: ApprovalContext
    env: SREEnvironment
    retriever: RunbookRetriever
    sensitive_agents: set[str]


def build_sre_environment(
    settings: Settings, scenario_path: str | Path
) -> SREEnvironment:
    """Pick the SRE backend. Local/CI → mock; prod → real HTTP (not implemented)."""
    if settings.app_env == "prod":
        raise NotImplementedError(
            "Real SRE integrations are not implemented; run with APP_ENV=local."
        )
    return MockSRE.from_scenario(scenario_path)


def build_retriever(
    settings: Settings,
    *,
    runbooks_dir: str | Path = DEFAULT_RUNBOOKS,
    embedder: Embedder | None = None,
    use_server: bool = False,
) -> RunbookRetriever:
    """Build and index the runbook retriever (in-memory Qdrant by default)."""
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
    client = (
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        if use_server
        else None
    )
    retriever = RunbookRetriever(
        embedder, collection=settings.qdrant_collection, client=client
    )
    retriever.index_directory(runbooks_dir)
    return retriever


def compute_sensitive_agents(
    tool_map: dict, registry: AgentRegistry
) -> set[str]:
    """Agents that own any tool requiring human approval (🟠/🔴)."""
    sensitive: set[str] = set()
    for name, spec in registry.agents.items():
        if any(
            tool_harm_tier(tool_map[t]).needs_human_approval
            for t in spec.tools
            if t in tool_map
        ):
            sensitive.add(name)
    return sensitive


def build_copilot(
    scenario_path: str | Path,
    *,
    settings: Settings | None = None,
    runbooks_dir: str | Path = DEFAULT_RUNBOOKS,
    embedder: Embedder | None = None,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    use_server: bool = False,
) -> IncidentCopilot:
    """Assemble the full incident copilot from settings and a scenario."""
    settings = settings or get_settings()
    registry = load_agent_registry()

    env = build_sre_environment(settings, scenario_path)
    retriever = build_retriever(
        settings, runbooks_dir=runbooks_dir, embedder=embedder, use_server=use_server
    )
    tool_map = build_tools(env, retriever)

    guard = PermissionGuard()
    approval_ctx = ApprovalContext()
    agents = build_all_agents(
        tool_map, guard, approval_ctx, settings=settings, model=model
    )
    sensitive = compute_sensitive_agents(tool_map, registry)

    graph = build_supervisor_graph(
        agents,
        workflow=registry.workflow,
        sensitive_agents=sensitive,
        approval_ctx=approval_ctx,
        checkpointer=checkpointer or MemorySaver(),
    )
    return IncidentCopilot(
        graph=graph,
        guard=guard,
        approval_ctx=approval_ctx,
        env=env,
        retriever=retriever,
        sensitive_agents=sensitive,
    )
