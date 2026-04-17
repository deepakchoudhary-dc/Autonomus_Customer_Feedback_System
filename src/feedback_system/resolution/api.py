from __future__ import annotations

from typing import Any, Protocol

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from feedback_system.agents.resolution_graph import build_resolution_graph
from feedback_system.resolution.schemas import ResolutionRequest, ResolutionResponse

router = APIRouter(prefix="/api/v1/resolution", tags=["resolution"])
logger = structlog.get_logger(__name__)


class AsyncResolutionGraph(Protocol):
    async def ainvoke(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run async graph invocation."""


def _is_escalated(result: dict[str, Any]) -> bool:
    return bool(result.get("hallucination_score") and int(result.get("retry_count", 0)) >= 3)


def get_resolution_graph(request: Request) -> AsyncResolutionGraph:
    graph = getattr(request.app.state, "resolution_graph", None)
    if graph is None:
        graph = build_resolution_graph()
        request.app.state.resolution_graph = graph
    return graph


@router.post(
    "/draft",
    response_model=ResolutionResponse,
    status_code=status.HTTP_200_OK,
)
async def draft_resolution(request: Request, payload: ResolutionRequest) -> ResolutionResponse:
    graph = get_resolution_graph(request)
    initial_state = {
        "customer_query": payload.customer_query,
        "retrieved_chunks": [],
        "draft_response": "",
        "hallucination_score": False,
        "retry_count": 0,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("resolution_graph_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Resolution agent unavailable",
        ) from exc

    return ResolutionResponse(
        draft_response=str(result.get("draft_response", "")),
        hallucination_score=bool(result.get("hallucination_score", False)),
        retry_count=int(result.get("retry_count", 0)),
        escalated=_is_escalated(result),
    )
