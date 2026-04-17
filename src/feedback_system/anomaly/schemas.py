from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackIngestedEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str = Field(min_length=1, max_length=128)
    customer_email: str = Field(min_length=3, max_length=320)
    source_platform: str = Field(min_length=1, max_length=64)
    raw_content: str = Field(min_length=1, max_length=10000)
    ingested_at: datetime | str

    @field_validator("customer_email")
    @classmethod
    def validate_customer_email(cls, value: str) -> str:
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            msg = "customer_email must be a valid email address"
            raise ValueError(msg)
        return value


class CriticalBugDetectedEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ai_summary: str
    root_cause_hypothesis: str
    affected_customer_ids: list[str]
    cluster_key: str
    trigger_count: int
    sample_ticket_ids: list[str]
