"""Autonomy policy: what each harm tier is allowed to do.

Maps ``HarmTier`` → ``PermissionDecision``. The guard (``guard.py``) uses this
table; LangGraph human-in-the-loop (Layer 6) resolves NEEDS_HUMAN_APPROVAL and
PROPOSE_ONLY by collecting approvals and feeding them back via ``ApprovalContext``.
"""

from __future__ import annotations

from enum import Enum

from app.domain.models import HarmTier


class PermissionDecision(str, Enum):
    """Outcome of a permission check before a tool runs."""

    ALLOW = "allow"                          # 🟢 execute immediately
    ALLOW_WITH_LOG = "allow_with_log"        # 🔵 execute + audit log
    NEEDS_CONFIRMATION = "needs_confirmation"  # 🟡 ask user/context first
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"  # 🟠 HITL interrupt
    PROPOSE_ONLY = "propose_only"            # 🔴 never auto-run; propose/escalate


def decision_for_tier(tier: HarmTier, *, is_confirmed: bool, is_approved: bool) -> PermissionDecision:
    """Return the decision for a tool call given its tier and approval state."""
    if tier is HarmTier.READ_ONLY:
        return PermissionDecision.ALLOW
    if tier is HarmTier.REVERSIBLE:
        return PermissionDecision.ALLOW_WITH_LOG
    if tier is HarmTier.COMPENSABLE:
        return PermissionDecision.ALLOW if is_confirmed else PermissionDecision.NEEDS_CONFIRMATION
    if tier is HarmTier.IRREVERSIBLE:
        return PermissionDecision.ALLOW if is_approved else PermissionDecision.NEEDS_HUMAN_APPROVAL
    if tier is HarmTier.CRITICAL:
        # Critical actions only run after explicit human approval; otherwise propose only.
        return PermissionDecision.ALLOW if is_approved else PermissionDecision.PROPOSE_ONLY
    return PermissionDecision.NEEDS_HUMAN_APPROVAL
