from pydantic import BaseModel, ConfigDict, Field


class ResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_query: str = Field(min_length=1, max_length=4000)


class ResolutionResponse(BaseModel):
    draft_response: str
    hallucination_score: bool
    retry_count: int
    escalated: bool
