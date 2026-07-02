"""Permission guard — enforces harm-tier autonomy before tool execution.

The guard sits between the agent and the tool (Layer 4). It reads each tool's
``harm_tier`` metadata (set in Layer 3), checks ``ApprovalContext``, and either
allows the call, blocks it with a clear message for the LLM, or records that
human approval is required (Layer 6 will pause the graph on that signal).

Usage::

    guard = PermissionGuard()
    ctx = ApprovalContext()
    result = guard.invoke(tools["query_logs"], {"service": "checkout", ...}, ctx)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import BaseTool

from app.security.policy import PermissionDecision, decision_for_tier
from app.tools.catalog import tool_harm_tier


def action_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    """Stable id for a specific tool invocation (used in approval sets)."""
    payload = json.dumps(args, sort_keys=True, default=str)
    return f"{tool_name}:{payload}"


@dataclass
class ApprovalContext:
    """Tracks confirmations/approvals for the current incident run."""

    confirmed: set[str] = field(default_factory=set)   # compensable tool fingerprints
    approved: set[str] = field(default_factory=set)    # irreversible/critical fingerprints


@dataclass
class AuditEntry:
    """One allow/deny/approval decision (for Langfuse / audit trail later)."""

    timestamp: datetime
    tool_name: str
    harm_tier: str
    decision: PermissionDecision
    fingerprint: str
    executed: bool
    detail: str = ""


class PermissionGuard:
    """Evaluate and optionally execute tool calls under harm-tier policy."""

    def __init__(self) -> None:
        self.audit_log: list[AuditEntry] = []

    def evaluate(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        ctx: ApprovalContext,
    ) -> PermissionDecision:
        tier = tool_harm_tier(tool)
        fp = action_fingerprint(tool.name, args)
        return decision_for_tier(
            tier,
            is_confirmed=fp in ctx.confirmed,
            is_approved=fp in ctx.approved,
        )

    def _record(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        decision: PermissionDecision,
        *,
        executed: bool,
        detail: str = "",
    ) -> None:
        self.audit_log.append(
            AuditEntry(
                timestamp=datetime.now(timezone.utc),
                tool_name=tool.name,
                harm_tier=tool_harm_tier(tool).value,
                decision=decision,
                fingerprint=action_fingerprint(tool.name, args),
                executed=executed,
                detail=detail,
            )
        )

    def invoke(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        ctx: ApprovalContext,
    ) -> str:
        """Check policy, then run the tool or return a block message for the LLM."""
        decision = self.evaluate(tool, args, ctx)
        tier = tool_harm_tier(tool)

        if decision is PermissionDecision.ALLOW:
            out = tool.invoke(args)
            self._record(tool, args, decision, executed=True)
            return str(out)

        if decision is PermissionDecision.ALLOW_WITH_LOG:
            out = tool.invoke(args)
            self._record(
                tool,
                args,
                decision,
                executed=True,
                detail="reversible action logged",
            )
            return str(out)

        if decision is PermissionDecision.NEEDS_CONFIRMATION:
            msg = (
                f"Tool '{tool.name}' ({tier.value}) requires contextual confirmation "
                f"before it can run. Add approval via ApprovalContext.confirm() and retry."
            )
            self._record(tool, args, decision, executed=False, detail=msg)
            return msg

        if decision is PermissionDecision.NEEDS_HUMAN_APPROVAL:
            msg = (
                f"Tool '{tool.name}' ({tier.value}) requires human approval before "
                f"execution. Escalate via human-in-the-loop (LangGraph interrupt)."
            )
            self._record(tool, args, decision, executed=False, detail=msg)
            return msg

        # PROPOSE_ONLY — critical tier without approval
        msg = (
            f"Tool '{tool.name}' ({tier.value}) cannot run autonomously. "
            f"Propose this action to a human operator for approval."
        )
        self._record(tool, args, decision, executed=False, detail=msg)
        return msg

    def confirm(self, tool: BaseTool, args: dict[str, Any], ctx: ApprovalContext) -> None:
        """Mark a compensable action as contextually confirmed."""
        ctx.confirmed.add(action_fingerprint(tool.name, args))

    def approve(self, tool: BaseTool, args: dict[str, Any], ctx: ApprovalContext) -> None:
        """Mark an irreversible/critical action as human-approved."""
        ctx.approved.add(action_fingerprint(tool.name, args))
