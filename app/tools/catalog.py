"""LangChain tools (Layer 3) — the agent's *use cases*.

Each tool wraps a repository call (Layer 2) and returns a compact, human-readable
string that an LLM can read. Two design points:

* **Harm tier as metadata.** Every tool carries ``metadata={"harm_tier": ...}``.
  The permission guard (Layer 4) reads this to decide the autonomy policy. Keeping
  the tier *on the tool* means one source of truth the whole system can inspect.
* **Dependency injection.** Tools close over an ``SREEnvironment`` + a
  ``RunbookRetriever`` passed into ``build_tools`` — so tests inject the mock and
  an offline retriever, while production injects the real ones. Tools never import
  a concrete backend.

Tools are intentionally "pure capability": they just perform the action. Gating a
dangerous action behind human approval is Layer 4's job, layered on top.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from app.domain.models import HarmTier, IncidentStatus
from app.integrations.protocols import SREEnvironment
from app.rag.retriever import RunbookRetriever

HARM_TIER_KEY = "harm_tier"


def tool_harm_tier(tool: BaseTool) -> HarmTier:
    """Read a tool's harm tier from its metadata (defaults to READ_ONLY)."""
    metadata = tool.metadata or {}
    return metadata.get(HARM_TIER_KEY, HarmTier.READ_ONLY)


def _make(func: object, harm_tier: HarmTier) -> StructuredTool:
    """Wrap a plain function as a StructuredTool, tagging its harm tier."""
    return StructuredTool.from_function(
        func,  # type: ignore[arg-type]
        metadata={HARM_TIER_KEY: harm_tier},
    )


# Agent → tool assignments live in app/agents/agents.yaml (dynamic registry).


def select(tools: dict[str, BaseTool], names: list[str]) -> list[BaseTool]:
    """Pick a subset of tools by name (used to equip each agent)."""
    return [tools[name] for name in names]


def build_tools(
    env: SREEnvironment, retriever: RunbookRetriever
) -> dict[str, BaseTool]:
    """Build every tool, wired to the given SRE environment and RAG retriever."""

    # ------------------------------------------------------------------ #
    # 🟢 read-only tools
    # ------------------------------------------------------------------ #

    def get_incident(incident_id: str) -> str:
        """Fetch an incident by id: severity, service, status, and description."""
        inc = env.get_incident(incident_id)
        if inc is None:
            return f"No incident found with id {incident_id}."
        return (
            f"{inc.id} [{inc.severity.value}] {inc.title} — service={inc.service}, "
            f"status={inc.status.value}. {inc.description}".strip()
        )

    def get_active_alerts() -> str:
        """List currently active monitoring alerts."""
        alerts = env.list_active_alerts()
        if not alerts:
            return "No active alerts."
        return "\n".join(f"- {a.id} [{a.severity.value}] {a.title}" for a in alerts)

    def query_logs(service: str, query: str, since_minutes: int = 60) -> str:
        """Search a service's logs for a substring within the last N minutes."""
        entries = env.query_logs(service, query, since_minutes)
        if not entries:
            return f"No logs matching '{query}' for {service} in last {since_minutes}m."
        return "\n".join(
            f"{e.timestamp:%H:%M} {e.level} {e.message}" for e in entries
        )

    def query_metrics(service: str, metric: str, window_minutes: int = 60) -> str:
        """Get a metric time series (e.g. error_rate, p99_latency_ms) for a service."""
        series = env.query_metrics(service, metric, window_minutes)
        if not series.points:
            return f"No data for metric '{metric}' on {service}."
        points = ", ".join(f"{p.value:g}{series.unit}" for p in series.points)
        return f"{metric} for {service} (last {window_minutes}m): {points}"

    def list_recent_deploys(service: str, limit: int = 5) -> str:
        """List recent deploys for a service, most recent first."""
        deploys = env.list_recent_deploys(service, limit)
        if not deploys:
            return f"No recent deploys for {service}."
        return "\n".join(
            f"- {d.id} {d.version} (commit {d.commit_sha}) status={d.status}"
            for d in deploys
        )

    def get_failing_ci_runs(service: str) -> str:
        """List failing CI runs for a service."""
        runs = env.get_failing_ci_runs(service)
        if not runs:
            return f"No failing CI runs for {service}."
        return "\n".join(
            f"- {r.id} {r.status} (commit {r.commit_sha}, branch {r.branch})"
            for r in runs
        )

    def get_ci_logs(run_id: str) -> str:
        """Fetch the logs of a CI run by id."""
        return env.get_ci_logs(run_id) or f"No logs for CI run {run_id}."

    def get_recent_commits(service: str, limit: int = 10) -> str:
        """List recent commits for a service (sha, author, message, files)."""
        commits = env.get_recent_commits(service, limit)
        if not commits:
            return f"No recent commits for {service}."
        return "\n".join(
            f"- {c.sha} {c.author}: {c.message} [{', '.join(c.files)}]"
            for c in commits
        )

    def git_blame(service: str, file: str) -> str:
        """Show the commits that touched a specific file."""
        commits = env.git_blame(service, file)
        if not commits:
            return f"No commits touched {file}."
        return "\n".join(f"- {c.sha} {c.author}: {c.message}" for c in commits)

    def search_runbooks(query: str) -> str:
        """Search runbooks and postmortems for guidance relevant to the query."""
        hits = retriever.search(query, k=3)
        if not hits:
            return "No relevant runbooks found."
        return "\n\n".join(f"## {h.title} ({h.source})\n{h.content}" for h in hits)

    # ------------------------------------------------------------------ #
    # write tools (increasing harm)
    # ------------------------------------------------------------------ #

    def post_incident_note(incident_id: str, text: str) -> str:
        """Post a note/comment to an incident (reversible)."""
        return env.post_note(incident_id, text).message

    def update_incident_status(incident_id: str, status: str) -> str:
        """Update an incident's status.

        Valid: open, investigating, identified, mitigated, resolved, escalated.
        """
        try:
            parsed = IncidentStatus(status)
        except ValueError:
            valid = [s.value for s in IncidentStatus]
            return f"Invalid status '{status}'. Valid values: {valid}"
        return env.update_status(incident_id, parsed).message

    def rerun_ci_job(run_id: str) -> str:
        """Re-trigger a CI job by run id (compensable)."""
        return env.rerun_ci_job(run_id).message

    def restart_service(service: str) -> str:
        """Restart a service (irreversible; requires human approval)."""
        return env.restart_service(service).message

    def scale_service(service: str, replicas: int) -> str:
        """Scale a service to N replicas (irreversible; requires human approval)."""
        return env.scale_service(service, replicas).message

    def rollback_deploy(service: str, to_version: str) -> str:
        """Roll a service back to a previous version (critical; approval only)."""
        return env.rollback_deploy(service, to_version).message

    def failover(service: str, region: str) -> str:
        """Fail a service over to another region (critical; approval only)."""
        return env.failover(service, region).message

    return {
        # 🟢 read-only
        "get_incident": _make(get_incident, HarmTier.READ_ONLY),
        "get_active_alerts": _make(get_active_alerts, HarmTier.READ_ONLY),
        "query_logs": _make(query_logs, HarmTier.READ_ONLY),
        "query_metrics": _make(query_metrics, HarmTier.READ_ONLY),
        "list_recent_deploys": _make(list_recent_deploys, HarmTier.READ_ONLY),
        "get_failing_ci_runs": _make(get_failing_ci_runs, HarmTier.READ_ONLY),
        "get_ci_logs": _make(get_ci_logs, HarmTier.READ_ONLY),
        "get_recent_commits": _make(get_recent_commits, HarmTier.READ_ONLY),
        "git_blame": _make(git_blame, HarmTier.READ_ONLY),
        "search_runbooks": _make(search_runbooks, HarmTier.READ_ONLY),
        # 🔵 reversible
        "post_incident_note": _make(post_incident_note, HarmTier.REVERSIBLE),
        "update_incident_status": _make(update_incident_status, HarmTier.REVERSIBLE),
        # 🟡 compensable
        "rerun_ci_job": _make(rerun_ci_job, HarmTier.COMPENSABLE),
        # 🟠 irreversible
        "restart_service": _make(restart_service, HarmTier.IRREVERSIBLE),
        "scale_service": _make(scale_service, HarmTier.IRREVERSIBLE),
        # 🔴 critical
        "rollback_deploy": _make(rollback_deploy, HarmTier.CRITICAL),
        "failover": _make(failover, HarmTier.CRITICAL),
    }
