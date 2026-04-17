import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from feedback_system.resolution.api import router


class FakeResolutionGraph:
    def __init__(self, should_fail: bool = False, escalated: bool = False) -> None:
        self.should_fail = should_fail
        self.escalated = escalated

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        if self.should_fail:
            raise RuntimeError("graph unavailable")

        if self.escalated:
            return {
                **state,
                "draft_response": "Escalated to a human support specialist due to unresolved grounding checks.",
                "hallucination_score": True,
                "retry_count": 3,
            }

        return {
            **state,
            "draft_response": "Please clear app cache and retry login.",
            "hallucination_score": False,
            "retry_count": 1,
        }


@pytest.mark.asyncio
async def test_resolution_draft_returns_response() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.resolution_graph = FakeResolutionGraph()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/resolution/draft",
            json={"customer_query": "I cannot log in"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["escalated"] is False
    assert payload["hallucination_score"] is False


@pytest.mark.asyncio
async def test_resolution_draft_reports_escalation() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.resolution_graph = FakeResolutionGraph(escalated=True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/resolution/draft",
            json={"customer_query": "My billing issue still persists"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["escalated"] is True
    assert payload["retry_count"] == 3


@pytest.mark.asyncio
async def test_resolution_draft_returns_503_on_graph_failure() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.resolution_graph = FakeResolutionGraph(should_fail=True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/resolution/draft",
            json={"customer_query": "Need refund"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Resolution agent unavailable"
