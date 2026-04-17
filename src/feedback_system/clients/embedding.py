try:
    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
except ModuleNotFoundError:
    APIConnectionError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception
    AsyncOpenAI = None
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from feedback_system.config import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed. Run `make install` first.")

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    async def create_embedding(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._model, input=text)
        return list(response.data[0].embedding)
