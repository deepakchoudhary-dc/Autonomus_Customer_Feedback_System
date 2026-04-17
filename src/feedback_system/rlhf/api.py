from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from feedback_system.config import get_settings
from feedback_system.rlhf.reward_model import save_reward_model, train_reward_model
from feedback_system.rlhf.schemas import (
    RLHFFeedbackAcceptedResponse,
    RLHFFeedbackPayload,
    RLHFTrainingResponse,
)
from feedback_system.rlhf.store import append_feedback_record, load_feedback_records

router = APIRouter(prefix="/api/v1/rlhf", tags=["rlhf"])


@router.post(
    "/feedback",
    response_model=RLHFFeedbackAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_human_feedback(request: Request, payload: RLHFFeedbackPayload) -> RLHFFeedbackAcceptedResponse:
    settings = get_settings()
    stored_samples = append_feedback_record(settings.rlhf_feedback_store_path, payload)

    try:
        await request.app.state.event_publisher.publish_event(
            settings.kafka_topic_rlhf_feedback,
            {
                "event_type": "RLHFFeedbackReceived",
                **payload.model_dump(mode="json"),
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to publish RLHF feedback event",
        ) from exc

    return RLHFFeedbackAcceptedResponse(status="accepted", stored_samples=stored_samples)


@router.post(
    "/train",
    response_model=RLHFTrainingResponse,
    status_code=status.HTTP_200_OK,
)
async def train_reward_model_now(request: Request) -> RLHFTrainingResponse:
    settings = get_settings()
    records = load_feedback_records(settings.rlhf_feedback_store_path)
    if len(records) < settings.rlhf_training_min_samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Insufficient RLHF samples to train reward model. "
                f"Need at least {settings.rlhf_training_min_samples}"
            ),
        )

    model_payload = train_reward_model(records)
    save_reward_model(model_payload, settings.rlhf_reward_model_path)

    await request.app.state.event_publisher.publish_event(
        settings.kafka_topic_rlhf_model_updates,
        {
            "event_type": "RLHFRewardModelUpdated",
            "sample_count": model_payload["sample_count"],
            "positive_ratio": round(float(model_payload["positive_ratio"]), 4),
            "reward_model_path": settings.rlhf_reward_model_path,
        },
    )

    return RLHFTrainingResponse(
        status="trained",
        sample_count=int(model_payload["sample_count"]),
        positive_ratio=round(float(model_payload["positive_ratio"]), 4),
    )
