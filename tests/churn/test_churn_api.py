import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from feedback_system.churn.api import router


class FakeChurnModel:
    def predict_probability(self, payload) -> float:  # noqa: ANN001
        return 0.87

    def risk_level(self, probability: float) -> str:
        return "high" if probability >= 0.8 else "medium"


@pytest.mark.asyncio
async def test_churn_predict_api_returns_score() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.churn_model = FakeChurnModel()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/churn/predict",
            json={
                "customer_id": "cust-77",
                "recent_negative_feedback_count": 3,
                "avg_sentiment_score": -0.5,
                "unresolved_ticket_count": 2,
                "avg_first_response_minutes": 55,
                "weekly_engagement_drop_ratio": 0.4,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "cust-77"
    assert body["risk_level"] == "high"
