"""Sub-agent factory — builds LangGraph ReAct agents from the YAML registry.

Each agent is a ``create_react_agent`` graph: an LLM ↔ tools loop until it finishes.
Agent definitions (prompt, tools, model tier) live in ``agents.yaml`` — add a new
agent there and call ``build_agent("new_name", ...)`` without rewiring this module.

Tools are wrapped by the permission guard so harm-tier policy is enforced on every call.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.agents.registry import AgentSpec, load_agent_registry
from app.config import Settings, get_settings
from app.gateway.client import get_chat_model
from app.security.guard import ApprovalContext, PermissionGuard


def guard_wrap(
    tool: BaseTool,
    guard: PermissionGuard,
    ctx: ApprovalContext,
) -> BaseTool:
    """Return a tool that routes every invocation through the permission guard."""

    def _run(**kwargs: Any) -> str:
        return guard.invoke(tool, kwargs, ctx)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        func=_run,
        args_schema=tool.args_schema,
        metadata=tool.metadata,
    )


def build_guarded_tools(
    tool_map: dict[str, BaseTool],
    tool_names: list[str],
    guard: PermissionGuard,
    ctx: ApprovalContext,
) -> list[BaseTool]:
    """Select tools by name and wrap each with the permission guard."""
    missing = [n for n in tool_names if n not in tool_map]
    if missing:
        raise KeyError(f"Unknown tool(s) in agent spec: {missing}")
    return [guard_wrap(tool_map[name], guard, ctx) for name in tool_names]


def get_agent_spec(name: str, *, registry_path: str | None = None) -> AgentSpec:
    """Look up one agent spec from the YAML registry."""
    registry = load_agent_registry(registry_path)
    if name not in registry.agents:
        known = ", ".join(sorted(registry.agents))
        raise KeyError(f"Unknown agent '{name}'. Known agents: {known}")
    return registry.agents[name]


def build_agent(
    name: str,
    tool_map: dict[str, BaseTool],
    guard: PermissionGuard,
    ctx: ApprovalContext,
    *,
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
    registry_path: str | None = None,
) -> CompiledStateGraph:
    """Build one ReAct sub-agent from the YAML registry."""
    spec = get_agent_spec(name, registry_path=registry_path)
    tools = build_guarded_tools(tool_map, spec.tools, guard, ctx)
    llm = model or get_chat_model(spec.model_tier, settings=settings or get_settings())
    return create_react_agent(
        llm,
        tools,
        prompt=spec.prompt,
        name=spec.name,
    )


def build_all_agents(
    tool_map: dict[str, BaseTool],
    guard: PermissionGuard,
    ctx: ApprovalContext,
    *,
    settings: Settings | None = None,
    registry_path: str | None = None,
) -> dict[str, CompiledStateGraph]:
    """Build every agent declared in the YAML registry (workflow order)."""
    registry = load_agent_registry(registry_path)
    names = registry.workflow or list(registry.agents.keys())
    return {
        name: build_agent(
            name,
            tool_map,
            guard,
            ctx,
            settings=settings,
            registry_path=registry_path,
        )
        for name in names
    }
