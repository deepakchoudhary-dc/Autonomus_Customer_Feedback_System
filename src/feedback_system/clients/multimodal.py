from __future__ import annotations

import asyncio
from dataclasses import dataclass

from feedback_system.config import Settings

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None


@dataclass(slots=True)
class MultimodalSynthesis:
    unified_content: str
    modalities: list[str]
    image_summaries: list[str]


class MultimodalUnderstandingClient:
    def __init__(self, settings: Settings) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed. Run `make install` first.")

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_multimodal_model

    async def _summarize_image_url(self, image_url: str) -> str:
        try:
            response = await self._client.responses.create(
                model=self._model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Describe only the issue-relevant details visible in this image. "
                                    "Return one short sentence."
                                ),
                            },
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )
        except Exception:
            return f"Image evidence submitted: {image_url}"

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return f"Image evidence submitted: {image_url}"

    async def synthesize_feedback(
        self,
        *,
        text_content: str | None,
        audio_transcript: str | None,
        video_transcript: str | None,
        image_urls: list[str],
    ) -> MultimodalSynthesis:
        tasks = [self._summarize_image_url(url) for url in image_urls]
        image_summaries = await asyncio.gather(*tasks) if tasks else []

        parts: list[str] = []
        modalities: list[str] = []

        if text_content:
            modalities.append("text")
            parts.append(f"Text signal: {text_content.strip()}")

        if audio_transcript:
            modalities.append("audio")
            parts.append(f"Audio transcript: {audio_transcript.strip()}")

        if video_transcript:
            modalities.append("video")
            parts.append(f"Video transcript: {video_transcript.strip()}")

        if image_summaries:
            modalities.append("image")
            for summary in image_summaries:
                parts.append(f"Image observation: {summary}")

        return MultimodalSynthesis(
            unified_content="\n".join(parts),
            modalities=modalities,
            image_summaries=image_summaries,
        )
