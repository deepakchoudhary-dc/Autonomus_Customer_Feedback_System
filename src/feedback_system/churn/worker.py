from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

try:
    from aiokafka import AIOKafkaConsumer
except ModuleNotFoundError:
    AIOKafkaConsumer = None

from feedback_system.churn.feature_store import CustomerFeatureStore
from feedback_system.churn.model import ChurnModel
from feedback_system.churn.schemas import ChurnAlertEvent, FeedbackSignalEvent
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
    publisher = EventPublisher(active_settings)
    model = ChurnModel(active_settings.churn_model_path)
    feature_store = CustomerFeatureStore()

    consumer = AIOKafkaConsumer(
        active_settings.kafka_topic_feedback_events,
        active_settings.kafka_topic_feedback_multimodal_events,
        bootstrap_servers=active_settings.kafka_bootstrap_servers,
        group_id=active_settings.kafka_consumer_group_churn_predictor,
        enable_auto_commit=False,
        value_deserializer=_kafka_value_deserializer,
    )

    await consumer.start()
    await publisher.start()
    logger.info("churn_worker_started")

    try:
        async for message in consumer:
            try:
                signal = FeedbackSignalEvent.model_validate(message.value)
                features = feature_store.update(signal)
                probability = model.predict_probability(features)

                if probability >= active_settings.churn_alert_threshold:
                    alert = ChurnAlertEvent(
                        event_type="ChurnRiskDetected",
                        customer_id=features.customer_id,
                        churn_probability=round(probability, 4),
                        risk_level=model.risk_level(probability),
                        explanation=(
                            "Detected sustained negative feedback and unresolved issue accumulation "
                            "for this customer."
                        ),
                        features={
                            "recent_negative_feedback_count": features.recent_negative_feedback_count,
                            "avg_sentiment_score": round(features.avg_sentiment_score, 4),
                            "unresolved_ticket_count": features.unresolved_ticket_count,
                            "avg_first_response_minutes": round(features.avg_first_response_minutes, 2),
                            "weekly_engagement_drop_ratio": round(
                                features.weekly_engagement_drop_ratio, 4
                            ),
                        },
                    )
                    await publisher.publish_event(
                        active_settings.kafka_topic_churn_alerts,
                        alert.model_dump(mode="json"),
                    )
                    logger.info(
                        "churn_alert_published",
                        customer_id=features.customer_id,
                        probability=round(probability, 4),
                    )

                await consumer.commit()
            except Exception as exc:
                logger.exception("churn_worker_message_failed", error=str(exc))
                await consumer.commit()
    finally:
        await consumer.stop()
        await publisher.stop()
        logger.info("churn_worker_stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
