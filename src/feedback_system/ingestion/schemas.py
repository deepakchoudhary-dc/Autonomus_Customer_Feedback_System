from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1, max_length=128)
    customer_email: str = Field(min_length=3, max_length=320)
    source_platform: str = Field(min_length=1, max_length=64)
    raw_content: str = Field(min_length=1, max_length=10000)

    @field_validator("customer_email")
    @classmethod
    def validate_customer_email(cls, value: str) -> str:
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            msg = "customer_email must be a valid email address"
            raise ValueError(msg)
        return value


class IngestionAcceptedResponse(BaseModel):
    status: str
    ticket_id: str


class MultimodalFeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1, max_length=128)
    customer_email: str = Field(min_length=3, max_length=320)
    source_platform: str = Field(min_length=1, max_length=64)
    text_content: str | None = Field(default=None, max_length=12000)
    audio_transcript: str | None = Field(default=None, max_length=12000)
    video_transcript: str | None = Field(default=None, max_length=12000)
    image_urls: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("customer_email")
    @classmethod
    def validate_multimodal_customer_email(cls, value: str) -> str:
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            msg = "customer_email must be a valid email address"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_at_least_one_modality(self) -> "MultimodalFeedbackPayload":
        has_text = bool(self.text_content and self.text_content.strip())
        has_audio = bool(self.audio_transcript and self.audio_transcript.strip())
        has_video = bool(self.video_transcript and self.video_transcript.strip())
        has_images = len(self.image_urls) > 0

        if not (has_text or has_audio or has_video or has_images):
            msg = (
                "At least one modality must be provided: text_content, audio_transcript, "
                "video_transcript, or image_urls"
            )
            raise ValueError(msg)
        return self


class MultimodalIngestionAcceptedResponse(BaseModel):
    status: str
    ticket_id: str
    vector_id: str
    modalities: list[str]
