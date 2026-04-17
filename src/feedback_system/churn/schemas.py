from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CustomerChurnFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=128)
    recent_negative_feedback_count: int = Field(ge=0)
    avg_sentiment_score: float = Field(ge=-1.0, le=1.0)
    unresolved_ticket_count: int = Field(ge=0)
    avg_first_response_minutes: float = Field(ge=0.0)
    weekly_engagement_drop_ratio: float = Field(ge=0.0, le=1.0)


class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str


class FeedbackSignalEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    customer_email: str
    raw_content: str
    ingested_at: datetime | str


class ChurnAlertEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    customer_id: str
    churn_probability: float
    risk_level: str
    explanation: str
    features: dict[str, float | int]
