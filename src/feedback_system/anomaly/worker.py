from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

try:
    from aiokafka import AIOKafkaConsumer
except ModuleNotFoundError:
    AIOKafkaConsumer = None

from feedback_system.anomaly.detector import FeedbackClusterDetector
from feedback_system.anomaly.schemas import FeedbackIngestedEvent
from feedback_system.config import Settings, get_settings
from feedback_system.events.publisher import EventPublisher

logger = structlog.get_logger(__name__)


def _kafka_value_deserializer(raw_value: bytes) -> dict[str, Any]:
    return json.loads(raw_value.decode("utf-8"))


async def run_worker(settings: Settings | None = None) -> None:
    if AIOKafkaConsumer is None:
        msg = "aiokafka package is not installed. Run `make install` first."
        raise RuntimeError(msg)

    active_settings = settings or get_settings()
    detector = FeedbackClusterDetector(threshold=active_settings.anomaly_cluster_threshold)
    publisher = EventPublisher(active_settings)

    consumer = AIOKafkaConsumer(
        active_settings.kafka_topic_feedback_events,
        active_settings.kafka_topic_feedback_multimodal_events,
        bootstrap_servers=active_settings.kafka_bootstrap_servers,
        group_id=active_settings.kafka_consumer_group_anomaly,
        enable_auto_commit=False,
        value_deserializer=_kafka_value_deserializer,
    )

    await consumer.start()
    await publisher.start()
    logger.info("anomaly_worker_started", topic=active_settings.kafka_topic_feedback_events)

    try:
        async for message in consumer:
            try:
                event = FeedbackIngestedEvent.model_validate(message.value)
                anomaly = detector.ingest_feedback(event)
                if anomaly is not None:
                    await publisher.publish_event(
                        active_settings.kafka_topic_critical_bugs,
                        anomaly.model_dump(mode="json"),
                    )
                    logger.info(
                        "critical_bug_identified",
                        cluster_key=anomaly.cluster_key,
                        trigger_count=anomaly.trigger_count,
                    )

                await consumer.commit()
            except Exception as exc:
                logger.exception("anomaly_worker_message_failed", error=str(exc))
                await consumer.commit()
    finally:
        await consumer.stop()
        await publisher.stop()
        logger.info("anomaly_worker_stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
