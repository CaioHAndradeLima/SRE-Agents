"""Sub-agent factory — builds LangGraph ReAct agents (Layer 5).

Each agent is a ``create_react_agent`` graph: an LLM ↔ tools loop until it finishes.
Tools are wrapped by the permission guard so harm-tier policy is enforced on every call.

Concepts introduced here:
* **ReAct agent** — the inner loop (LLM reasons → calls tool → observes result → repeats).
* **Dynamic specialization** — same factory, different tool set + prompt + model tier per role.
* **Guarded tools** — Layer 4 sits between the agent and raw tools transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from app.agents.prompts import DIAGNOSIS_PROMPT, REMEDIATION_PROMPT, TRIAGE_PROMPT
from app.config import Settings, get_settings
from app.gateway.client import ModelTier, get_chat_model
from app.security.guard import ApprovalContext, PermissionGuard
from app.tools.catalog import (
    DIAGNOSIS_TOOLS,
    REMEDIATION_TOOLS,
    TRIAGE_TOOLS,
)

AgentName = Literal["triage", "diagnosis", "remediation"]


@dataclass(frozen=True)
class AgentSpec:
    """Definition of a specialized sub-agent (registry entry)."""

    name: AgentName
    tool_names: list[str]
    model_tier: ModelTier
    prompt: str


AGENT_REGISTRY: dict[AgentName, AgentSpec] = {
    "triage": AgentSpec(
        name="triage",
        tool_names=TRIAGE_TOOLS,
        model_tier="fast",
        prompt=TRIAGE_PROMPT,
    ),
    "diagnosis": AgentSpec(
        name="diagnosis",
        tool_names=DIAGNOSIS_TOOLS,
        model_tier="reasoning",
        prompt=DIAGNOSIS_PROMPT,
    ),
    "remediation": AgentSpec(
        name="remediation",
        tool_names=REMEDIATION_TOOLS,
        model_tier="reasoning",
        prompt=REMEDIATION_PROMPT,
    ),
}


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
    return [
        guard_wrap(tool_map[name], guard, ctx)
        for name in tool_names
    ]


def build_agent(
    name: AgentName,
    tool_map: dict[str, BaseTool],
    guard: PermissionGuard,
    ctx: ApprovalContext,
    *,
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build one ReAct sub-agent for the given role."""
    spec = AGENT_REGISTRY[name]
    tools = build_guarded_tools(tool_map, spec.tool_names, guard, ctx)
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
) -> dict[AgentName, CompiledStateGraph]:
    """Build the lean v1 team: Triage, Diagnosis, Remediation."""
    return {
        name: build_agent(name, tool_map, guard, ctx, settings=settings)
        for name in ("triage", "diagnosis", "remediation")
    }
