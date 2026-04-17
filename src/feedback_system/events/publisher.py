import json
from typing import Any

try:
    from aiokafka import AIOKafkaProducer
except ModuleNotFoundError:
    AIOKafkaProducer = None

from feedback_system.config import Settings


class EventPublisher:
    def __init__(self, settings: Settings) -> None:
        if AIOKafkaProducer is None:
            raise RuntimeError("aiokafka package is not installed. Run `make install` first.")

        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        self._topic = settings.kafka_topic_feedback_events
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return

        await self._producer.stop()
        self._started = False

    async def publish_event(self, topic: str, payload: dict[str, Any]) -> None:
        if not self._started:
            await self.start()

        await self._producer.send_and_wait(topic, payload)

    async def publish_feedback_ingested(self, payload: dict[str, Any]) -> None:
        await self.publish_event(self._topic, payload)
