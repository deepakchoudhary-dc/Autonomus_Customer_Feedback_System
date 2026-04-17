from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from feedback_system.ingestion.api import router


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embedding: list[float] = [0.1, 0.2, 0.3]
        self.should_fail = False
        self.last_text = ""

    async def create_embedding(self, text: str) -> list[float]:
        self.last_text = text
        if self.should_fail:
            raise RuntimeError("embedding error")
        return self.embedding


class FakeVectorStoreClient:
    def __init__(self) -> None:
        self.should_fail = False
        self.last_payload: dict[str, Any] | None = None

    async def upsert_feedback_vector(
        self,
        *,
        ticket_id: str,
        source_platform: str,
        customer_email: str,
        embedding: list[float],
        vector_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.should_fail:
            raise RuntimeError("vector error")
        self.last_payload = {
            "ticket_id": ticket_id,
            "source_platform": source_platform,
            "customer_email": customer_email,
            "embedding": embedding,
            "vector_id": vector_id,
            "extra_metadata": extra_metadata,
        }


class FakeMultimodalClient:
    def __init__(self) -> None:
        self.should_fail = False

    async def synthesize_feedback(
        self,
        *,
        text_content: str | None,
        audio_transcript: str | None,
        video_transcript: str | None,
        image_urls: list[str],
    ) -> Any:
        if self.should_fail:
            raise RuntimeError("multimodal error")

        parts = [part for part in [text_content, audio_transcript, video_transcript] if part]
        parts.extend(f"image:{url}" for url in image_urls)
        return type(
            "Synthesis",
            (),
            {
                "unified_content": " | ".join(parts),
                "modalities": ["text", "audio", "video", "image"],
                "image_summaries": [f"summary:{url}" for url in image_urls],
            },
        )


class FakeEventPublisher:
    def __init__(self) -> None:
        self.should_fail = False
        self.last_event: dict[str, Any] | None = None
        self.last_topic: str | None = None

    async def publish_feedback_ingested(self, payload: dict[str, Any]) -> None:
        if self.should_fail:
            raise RuntimeError("event error")
        self.last_event = payload

    async def publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        if self.should_fail:
            raise RuntimeError("event error")
        self.last_topic = topic
        self.last_event = payload


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.embedding_client = FakeEmbeddingClient()
    app.state.multimodal_understanding_client = FakeMultimodalClient()
    app.state.vector_store_client = FakeVectorStoreClient()
    app.state.event_publisher = FakeEventPublisher()
    return app


async def post_feedback(app: FastAPI, payload: dict[str, Any]) -> Any:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/api/v1/webhooks/feedback", json=payload)


async def post_multimodal_feedback(app: FastAPI, payload: dict[str, Any]) -> Any:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/api/v1/webhooks/feedback-multimodal", json=payload)


@pytest.mark.asyncio
async def test_feedback_ingestion_returns_202(app: FastAPI) -> None:
    payload = {
        "ticket_id": "TICKET-100",
        "customer_email": "user@example.com",
        "source_platform": "zendesk",
        "raw_content": "I am unable to complete checkout.",
    }

    response = await post_feedback(app, payload)

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert app.state.embedding_client.last_text == payload["raw_content"]
    assert app.state.vector_store_client.last_payload is not None
    assert app.state.vector_store_client.last_payload["ticket_id"] == payload["ticket_id"]
    assert app.state.vector_store_client.last_payload["extra_metadata"]["content_type"] == "text"
    assert app.state.event_publisher.last_event is not None
    assert app.state.event_publisher.last_event["event_type"] == "FeedbackIngested"


@pytest.mark.asyncio
async def test_feedback_ingestion_rejects_invalid_payload(app: FastAPI) -> None:
    payload = {
        "ticket_id": "",
        "customer_email": "not-an-email",
        "source_platform": "",
        "raw_content": "",
    }

    response = await post_feedback(app, payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_feedback_ingestion_returns_503_when_embedding_fails(app: FastAPI) -> None:
    app.state.embedding_client.should_fail = True
    payload = {
        "ticket_id": "TICKET-101",
        "customer_email": "user@example.com",
        "source_platform": "intercom",
        "raw_content": "Billing amount is incorrect.",
    }

    response = await post_feedback(app, payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service unavailable"


@pytest.mark.asyncio
async def test_feedback_ingestion_returns_503_when_vector_upsert_fails(app: FastAPI) -> None:
    app.state.vector_store_client.should_fail = True
    payload = {
        "ticket_id": "TICKET-102",
        "customer_email": "user@example.com",
        "source_platform": "form",
        "raw_content": "App crashes on login.",
    }

    response = await post_feedback(app, payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Vector storage unavailable"


@pytest.mark.asyncio
async def test_feedback_ingestion_returns_503_when_publish_fails(app: FastAPI) -> None:
    app.state.event_publisher.should_fail = True
    payload = {
        "ticket_id": "TICKET-103",
        "customer_email": "user@example.com",
        "source_platform": "whatsapp",
        "raw_content": "Refund has not been processed.",
    }

    response = await post_feedback(app, payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Event bus unavailable"


@pytest.mark.asyncio
async def test_multimodal_feedback_ingestion_returns_202(app: FastAPI) -> None:
    payload = {
        "ticket_id": "TICKET-201",
        "customer_email": "user@example.com",
        "source_platform": "mobile-app",
        "text_content": "App freezes during payment",
        "audio_transcript": "I cannot complete checkout",
        "video_transcript": "Screen stuck on loading",
        "image_urls": ["https://example.com/screenshot.png"],
    }

    response = await post_multimodal_feedback(app, payload)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["ticket_id"] == payload["ticket_id"]
    assert "vector_id" in body
    assert app.state.vector_store_client.last_payload is not None
    assert app.state.vector_store_client.last_payload["extra_metadata"]["content_type"] == "multimodal"


@pytest.mark.asyncio
async def test_multimodal_feedback_requires_at_least_one_signal(app: FastAPI) -> None:
    payload = {
        "ticket_id": "TICKET-202",
        "customer_email": "user@example.com",
        "source_platform": "mobile-app",
        "text_content": None,
        "audio_transcript": None,
        "video_transcript": None,
        "image_urls": [],
    }

    response = await post_multimodal_feedback(app, payload)

    assert response.status_code == 422
