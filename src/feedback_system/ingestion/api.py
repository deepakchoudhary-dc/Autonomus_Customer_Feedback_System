from datetime import datetime, timezone
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from feedback_system.clients.embedding import EmbeddingClient
from feedback_system.clients.multimodal import MultimodalUnderstandingClient
from feedback_system.clients.vector_store import VectorStoreClient
from feedback_system.config import get_settings
from feedback_system.events.publisher import EventPublisher
from feedback_system.ingestion.schemas import (
    FeedbackPayload,
    IngestionAcceptedResponse,
    MultimodalFeedbackPayload,
    MultimodalIngestionAcceptedResponse,
)

router = APIRouter(prefix="/api/v1/webhooks", tags=["ingestion"])
logger = structlog.get_logger(__name__)


def get_embedding_client(request: Request) -> EmbeddingClient:
    return request.app.state.embedding_client


def get_vector_store_client(request: Request) -> VectorStoreClient:
    return request.app.state.vector_store_client


def get_event_publisher(request: Request) -> EventPublisher:
    return request.app.state.event_publisher


def get_multimodal_understanding_client(request: Request) -> MultimodalUnderstandingClient:
    return request.app.state.multimodal_understanding_client


@router.post(
    "/feedback",
    response_model=IngestionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_feedback(
    payload: FeedbackPayload,
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    vector_store_client: Annotated[VectorStoreClient, Depends(get_vector_store_client)],
    event_publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> IngestionAcceptedResponse:
    openai_start = perf_counter()

    try:
        embedding = await embedding_client.create_embedding(payload.raw_content)
    except Exception as exc:
        logger.exception(
            "embedding_generation_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        ) from exc

    latency_ms = round((perf_counter() - openai_start) * 1000, 2)
    logger.info("embedding_generated", ticket_id=payload.ticket_id, latency_ms=latency_ms)

    try:
        await vector_store_client.upsert_feedback_vector(
            ticket_id=payload.ticket_id,
            source_platform=payload.source_platform,
            customer_email=str(payload.customer_email),
            embedding=embedding,
            extra_metadata={
                "raw_content": payload.raw_content,
                "content_type": "text",
            },
        )
    except Exception as exc:
        logger.exception(
            "pinecone_upsert_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage unavailable",
        ) from exc

    event_payload = {
        "event_type": "FeedbackIngested",
        "ticket_id": payload.ticket_id,
        "customer_email": str(payload.customer_email),
        "source_platform": payload.source_platform,
        "raw_content": payload.raw_content,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await event_publisher.publish_feedback_ingested(event_payload)
    except Exception as exc:
        logger.exception(
            "kafka_publish_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event bus unavailable",
        ) from exc

    logger.info(
        "feedback_ingested",
        ticket_id=payload.ticket_id,
        source_platform=payload.source_platform,
    )
    return IngestionAcceptedResponse(status="accepted", ticket_id=payload.ticket_id)


@router.post(
    "/feedback-multimodal",
    response_model=MultimodalIngestionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_feedback_multimodal(
    payload: MultimodalFeedbackPayload,
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    multimodal_client: Annotated[
        MultimodalUnderstandingClient, Depends(get_multimodal_understanding_client)
    ],
    vector_store_client: Annotated[VectorStoreClient, Depends(get_vector_store_client)],
    event_publisher: Annotated[EventPublisher, Depends(get_event_publisher)],
) -> MultimodalIngestionAcceptedResponse:
    settings = get_settings()

    try:
        synthesis = await multimodal_client.synthesize_feedback(
            text_content=payload.text_content,
            audio_transcript=payload.audio_transcript,
            video_transcript=payload.video_transcript,
            image_urls=payload.image_urls,
        )
    except Exception as exc:
        logger.exception("multimodal_synthesis_failed", ticket_id=payload.ticket_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal understanding unavailable",
        ) from exc

    if not synthesis.unified_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to synthesize multimodal content",
        )

    openai_start = perf_counter()
    try:
        embedding = await embedding_client.create_embedding(synthesis.unified_content)
    except Exception as exc:
        logger.exception(
            "multimodal_embedding_generation_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        ) from exc

    latency_ms = round((perf_counter() - openai_start) * 1000, 2)
    vector_id = f"{payload.ticket_id}-mm-{uuid4().hex[:10]}"

    try:
        await vector_store_client.upsert_feedback_vector(
            ticket_id=payload.ticket_id,
            source_platform=payload.source_platform,
            customer_email=payload.customer_email,
            embedding=embedding,
            vector_id=vector_id,
            extra_metadata={
                "raw_content": synthesis.unified_content,
                "content_type": "multimodal",
                "modalities": ",".join(synthesis.modalities),
                "image_count": len(payload.image_urls),
            },
        )
    except Exception as exc:
        logger.exception(
            "multimodal_pinecone_upsert_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector storage unavailable",
        ) from exc

    event_payload = {
        "event_type": "FeedbackIngestedMultimodal",
        "ticket_id": payload.ticket_id,
        "vector_id": vector_id,
        "customer_email": payload.customer_email,
        "source_platform": payload.source_platform,
        "raw_content": synthesis.unified_content,
        "modalities": synthesis.modalities,
        "image_summaries": synthesis.image_summaries,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await event_publisher.publish_event(settings.kafka_topic_feedback_multimodal_events, event_payload)
        await event_publisher.publish_feedback_ingested(event_payload)
    except Exception as exc:
        logger.exception(
            "multimodal_kafka_publish_failed",
            ticket_id=payload.ticket_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event bus unavailable",
        ) from exc

    logger.info(
        "multimodal_feedback_ingested",
        ticket_id=payload.ticket_id,
        source_platform=payload.source_platform,
        vector_id=vector_id,
        modalities=synthesis.modalities,
        embedding_latency_ms=latency_ms,
    )

    return MultimodalIngestionAcceptedResponse(
        status="accepted",
        ticket_id=payload.ticket_id,
        vector_id=vector_id,
        modalities=synthesis.modalities,
    )
