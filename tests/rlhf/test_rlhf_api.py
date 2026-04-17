import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from feedback_system.config import Settings
from feedback_system.rlhf.api import router


class FakeEventPublisher:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish_event(self, topic: str, payload: dict) -> None:  # noqa: ANN001
        self.events.append((topic, payload))


@pytest.mark.asyncio
async def test_rlhf_feedback_endpoint_and_train_endpoint(monkeypatch, tmp_path) -> None:
    feedback_path = tmp_path / "rlhf_feedback.jsonl"
    reward_model_path = tmp_path / "reward_model.json"

    settings = Settings(
        rlhf_feedback_store_path=str(feedback_path),
        rlhf_reward_model_path=str(reward_model_path),
        rlhf_training_min_samples=2,
        kafka_topic_rlhf_feedback="rlhf-feedback-events",
        kafka_topic_rlhf_model_updates="rlhf-model-updates",
    )
    monkeypatch.setattr("feedback_system.rlhf.api.get_settings", lambda: settings)

    app = FastAPI()
    app.include_router(router)
    app.state.event_publisher = FakeEventPublisher()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response_one = await client.post(
            "/api/v1/rlhf/feedback",
            json={
                "prompt": "How to cancel subscription?",
                "response": "Open billing and choose cancel plan.",
                "retrieved_context": ["Billing page has cancel option"],
                "rating": 1,
                "reviewer_id": "reviewer-1",
            },
        )
        response_two = await client.post(
            "/api/v1/rlhf/feedback",
            json={
                "prompt": "How to cancel subscription?",
                "response": "This is not supported.",
                "retrieved_context": ["Billing page has cancel option"],
                "rating": -1,
                "reviewer_id": "reviewer-2",
            },
        )
        train_response = await client.post("/api/v1/rlhf/train")

    assert response_one.status_code == 202
    assert response_two.status_code == 202
    assert train_response.status_code == 200
    assert train_response.json()["status"] == "trained"

    assert reward_model_path.exists()
    payload = json.loads(reward_model_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 2
