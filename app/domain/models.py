"""Domain models — the shared vocabulary of the whole system.

These are plain pydantic types with **no dependency on LangChain/LangGraph**. Every
other layer (repositories, tools, agents, graph) imports from here, so this file
defines the nouns the application speaks: incidents, their severity/status, the
harm tier of an action, and the standard result every tool returns.

Keeping this layer pure means it is trivially unit-testable and never needs an LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Timezone-aware 'now' (avoids naive datetimes in serialized output)."""
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    """Incident severity, CRITICAL (worst) → LOW (lowest)."""

    CRITICAL = "CRITICAL"  # major outage / data loss
    HIGH = "HIGH"  # significant degradation
    MEDIUM = "MEDIUM"  # minor / partial impact
    LOW = "LOW"  # cosmetic / informational


class IncidentStatus(str, Enum):
    """Lifecycle of an incident as the copilot works it."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"     # root cause hypothesized
    MITIGATED = "mitigated"       # fix applied, verifying
    RESOLVED = "resolved"
    ESCALATED = "escalated"       # handed to a human


class HarmTier(str, Enum):
    """How much damage a tool call can do — the axis our security model uses.

    Ordered from least to most dangerous. The numeric ``level`` lets the
    permission guard (Layer 4) compare tiers and decide the required autonomy
    policy (auto / log / confirm / human-approval / forbidden).
    """

    READ_ONLY = "read_only"        # search, list, analyze
    REVERSIBLE = "reversible"      # create a draft, update a recoverable field
    COMPENSABLE = "compensable"    # action that can still be corrected
    IRREVERSIBLE = "irreversible"  # delete, restart, scale
    CRITICAL = "critical"          # rollback prod, failover, data loss

    @property
    def level(self) -> int:
        """0 (read-only) … 4 (critical); higher = more dangerous."""
        return _HARM_ORDER.index(self)

    @property
    def needs_human_approval(self) -> bool:
        """True for tiers dangerous enough to require a human in the loop.

        Derived from the ordering (single source of truth): everything at
        IRREVERSIBLE or above must be approved by a human before it runs. The
        finer policy for CRITICAL (propose-only / never auto-execute) is enforced
        by the permission guard in Layer 4.
        """
        return self.level >= HarmTier.IRREVERSIBLE.level

    def __lt__(self, other: object) -> bool:  # enables sorting / comparisons
        if isinstance(other, HarmTier):
            return self.level < other.level
        return NotImplemented


# Single source of truth for harm-tier ordering.
_HARM_ORDER: tuple[HarmTier, ...] = (
    HarmTier.READ_ONLY,
    HarmTier.REVERSIBLE,
    HarmTier.COMPENSABLE,
    HarmTier.IRREVERSIBLE,
    HarmTier.CRITICAL,
)


class Incident(BaseModel):
    """A CI/production incident the copilot is asked to resolve."""

    id: str
    title: str
    service: str
    severity: Severity = Severity.MEDIUM
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class ToolResult(BaseModel):
    """Standard envelope every tool returns.

    A uniform shape makes agents easier to reason about and tests simpler: instead
    of each tool returning ad-hoc strings/dicts, callers always get ``success`` +
    a human-readable ``summary`` (what the LLM reads) + structured ``data``.
    """

    success: bool = True
    summary: str = ""
    data: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def ok(cls, summary: str, **data: Any) -> ToolResult:
        """Convenience constructor for a successful result."""
        return cls(success=True, summary=summary, data=data or None)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        """Convenience constructor for a failed result."""
        return cls(success=False, summary=error, error=error)
