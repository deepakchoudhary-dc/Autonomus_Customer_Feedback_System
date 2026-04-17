from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

try:
    from aiokafka import AIOKafkaConsumer
except ModuleNotFoundError:
    AIOKafkaConsumer = None

from feedback_system.config import Settings, get_settings
from feedback_system.events.publisher import EventPublisher

logger = structlog.get_logger(__name__)


class JiraIssueResolvedEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issue_key: str = Field(min_length=1)
    resolution_summary: str = Field(min_length=3)
    affected_customer_ids: list[str] = Field(default_factory=list)


def build_customer_notification(event: JiraIssueResolvedEvent, customer_id: str) -> dict[str, Any]:
    return {
        "event_type": "CustomerIssueResolved",
        "customer_id": customer_id,
        "issue_key": event.issue_key,
        "message": (
            "Your reported issue has been resolved. "
            f"Reference: {event.issue_key}. "
            f"Summary: {event.resolution_summary}"
        ),
    }


def _kafka_value_deserializer(raw_value: bytes) -> dict[str, Any]:
    return json.loads(raw_value.decode("utf-8"))


async def run_worker(settings: Settings | None = None) -> None:
    if AIOKafkaConsumer is None:
        msg = "aiokafka package is not installed. Run `make install` first."
        raise RuntimeError(msg)

    active_settings = settings or get_settings()
    publisher = EventPublisher(active_settings)
    consumer = AIOKafkaConsumer(
        active_settings.kafka_topic_jira_resolved,
        bootstrap_servers=active_settings.kafka_bootstrap_servers,
        group_id=active_settings.kafka_consumer_group_resolution_notifier,
        enable_auto_commit=False,
        value_deserializer=_kafka_value_deserializer,
    )

    await consumer.start()
    await publisher.start()
    logger.info("resolution_notifier_started", topic=active_settings.kafka_topic_jira_resolved)

    try:
        async for message in consumer:
            try:
                event = JiraIssueResolvedEvent.model_validate(message.value)
                for customer_id in event.affected_customer_ids:
                    notification = build_customer_notification(event, customer_id)
                    await publisher.publish_event(
                        active_settings.kafka_topic_customer_notifications,
                        notification,
                    )

                await consumer.commit()
                logger.info(
                    "resolution_notifications_published",
                    issue_key=event.issue_key,
                    recipients=len(event.affected_customer_ids),
                )
            except Exception as exc:
                logger.exception("resolution_notifier_message_failed", error=str(exc))
                await consumer.commit()
    finally:
        await consumer.stop()
        await publisher.stop()
        logger.info("resolution_notifier_stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
