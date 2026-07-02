"""Load the dynamic agent registry from YAML.

Adding a new specialized agent = add an entry to ``agents.yaml`` (tools + prompt +
model tier). No factory rewiring required — ``build_agent(spec.name, ...)`` picks it
up at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from app.gateway.client import ModelTier

DEFAULT_AGENTS_YAML = Path(__file__).resolve().parent / "agents.yaml"


class AgentSpec(BaseModel):
    """One specialized sub-agent definition (from YAML)."""

    name: str
    model_tier: ModelTier
    tools: list[str] = Field(min_length=1)
    prompt: str = Field(min_length=1)


class AgentRegistry(BaseModel):
    """Full registry: agents keyed by name + optional workflow order."""

    agents: dict[str, AgentSpec]
    workflow: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def workflow_refs_known_agents(self) -> AgentRegistry:
        unknown = [name for name in self.workflow if name not in self.agents]
        if unknown:
            raise ValueError(f"workflow references unknown agents: {unknown}")
        return self


def _parse_registry(data: dict) -> AgentRegistry:
    raw_agents = data.get("agents") or {}
    agents = {
        name: AgentSpec(name=name, **spec)
        for name, spec in raw_agents.items()
    }
    return AgentRegistry(agents=agents, workflow=data.get("workflow") or list(agents))


@lru_cache
def load_agent_registry(path: str | None = None) -> AgentRegistry:
    """Load and validate the agent registry from YAML (cached by path)."""
    yaml_path = Path(path) if path else DEFAULT_AGENTS_YAML
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return _parse_registry(data)


def list_agent_names(path: str | None = None) -> list[str]:
    """Return agent names in workflow order (or dict order if workflow empty)."""
    reg = load_agent_registry(path)
    return reg.workflow or list(reg.agents.keys())
