from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RLHFFeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=16000)
    response: str = Field(min_length=1, max_length=16000)
    retrieved_context: list[str] = Field(default_factory=list, max_length=20)
    rating: int = Field(ge=-1, le=1)
    reviewer_id: str = Field(min_length=1, max_length=128)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("rating")
    @classmethod
    def validate_binary_feedback(cls, value: int) -> int:
        if value == 0:
            msg = "rating cannot be 0; use +1 for preferred and -1 for rejected"
            raise ValueError(msg)
        return value


class RLHFFeedbackAcceptedResponse(BaseModel):
    status: str
    stored_samples: int


class RLHFTrainingResponse(BaseModel):
    status: str
    sample_count: int
    positive_ratio: float


class RewardModelUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    sample_count: int
    positive_ratio: float
    reward_model_path: str
