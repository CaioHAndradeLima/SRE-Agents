"""FastAPI app — interactive entry point for the SRE incident copilot.

Reuses the composition root untouched: the API is a thin driver over
:class:`IncidentService`. Run it with::

    uvicorn app.api.server:app --reload

Endpoints:
    GET  /healthz                      liveness
    GET  /scenarios                    list seeded incidents you can run
    POST /incidents                    start an incident from a scenario
    GET  /incidents/{id}               current status / findings / audit
    POST /incidents/{id}/approve       approve or reject a paused high-risk step
    GET  /incidents/{id}/audit         full audit trail
    POST /incidents/{id}/eval          grade the run with the LLM-as-judge
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request

from app.api.schemas import (
    ApprovalRequest,
    AuditItem,
    EvalResponse,
    IncidentState,
    ScenarioSummary,
    StartIncidentRequest,
)
from app.api.service import (
    IncidentService,
    InvalidApprovalState,
    ScenarioNotFound,
    SessionNotFound,
)
from app.config import get_settings
from app.observability.tracing import setup_observability


def get_service(request: Request) -> IncidentService:
    """Resolve the app-scoped incident service (overridable in tests)."""
    return request.app.state.service


def create_app(service: IncidentService | None = None) -> FastAPI:
    """Build the FastAPI app, wiring a (possibly injected) incident service."""
    app = FastAPI(
        title="SRE Incident Copilot",
        version="1.0.0",
        summary="Drive a multi-agent SRE copilot over seeded incident scenarios.",
    )

    if service is None:
        settings = get_settings()
        setup_observability(settings)
        service = IncidentService(settings=settings)
    app.state.service = service

    @app.exception_handler(ScenarioNotFound)
    async def _scenario_not_found(_: Request, exc: ScenarioNotFound):
        raise HTTPException(status_code=404, detail=f"scenario not found: {exc}")

    @app.exception_handler(SessionNotFound)
    async def _session_not_found(_: Request, exc: SessionNotFound):
        raise HTTPException(status_code=404, detail=f"incident session not found: {exc}")

    @app.exception_handler(InvalidApprovalState)
    async def _invalid_approval(_: Request, exc: InvalidApprovalState):
        raise HTTPException(
            status_code=409, detail="session is not awaiting approval"
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/scenarios", response_model=list[ScenarioSummary])
    async def list_scenarios(
        svc: IncidentService = Depends(get_service),
    ) -> list[ScenarioSummary]:
        return svc.list_scenarios()

    @app.post("/incidents", response_model=IncidentState)
    async def start_incident(
        body: StartIncidentRequest,
        svc: IncidentService = Depends(get_service),
    ) -> IncidentState:
        session = svc.start(body.scenario_id, prompt=body.prompt)
        return svc.snapshot(session)

    @app.get("/incidents/{session_id}", response_model=IncidentState)
    async def get_incident(
        session_id: str,
        svc: IncidentService = Depends(get_service),
    ) -> IncidentState:
        return svc.snapshot(svc.get_session(session_id))

    @app.post("/incidents/{session_id}/approve", response_model=IncidentState)
    async def approve_incident(
        session_id: str,
        body: ApprovalRequest,
        svc: IncidentService = Depends(get_service),
    ) -> IncidentState:
        session = svc.approve(session_id, body.approved)
        return svc.snapshot(session)

    @app.get("/incidents/{session_id}/audit", response_model=list[AuditItem])
    async def get_audit(
        session_id: str,
        svc: IncidentService = Depends(get_service),
    ) -> list[AuditItem]:
        return svc.snapshot(svc.get_session(session_id)).audit

    @app.post("/incidents/{session_id}/eval", response_model=EvalResponse)
    async def eval_incident(
        session_id: str,
        svc: IncidentService = Depends(get_service),
    ) -> EvalResponse:
        return svc.evaluate(session_id)

    return app


app = create_app()
