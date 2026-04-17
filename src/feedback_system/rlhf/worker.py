from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

try:
    from aiokafka import AIOKafkaConsumer
except ModuleNotFoundError:
    AIOKafkaConsumer = None

from feedback_system.config import Settings, get_settings
from feedback_system.events.publisher import EventPublisher
from feedback_system.rlhf.reward_model import save_reward_model, train_reward_model
from feedback_system.rlhf.schemas import RLHFFeedbackPayload
from feedback_system.rlhf.store import append_feedback_record, count_feedback_records, load_feedback_records

logger = structlog.get_logger(__name__)


def _kafka_value_deserializer(raw_value: bytes) -> dict[str, Any]:
    return json.loads(raw_value.decode("utf-8"))


async def run_worker(settings: Settings | None = None) -> None:
    if AIOKafkaConsumer is None:
        msg = "aiokafka package is not installed. Run `make install` first."
        raise RuntimeError(msg)

    active_settings = settings or get_settings()
    publisher = EventPublisher(active_settings)
    pending_since_last_train = 0

    consumer = AIOKafkaConsumer(
        active_settings.kafka_topic_rlhf_feedback,
        bootstrap_servers=active_settings.kafka_bootstrap_servers,
        group_id=active_settings.kafka_consumer_group_rlhf,
        enable_auto_commit=False,
        value_deserializer=_kafka_value_deserializer,
    )

    await consumer.start()
    await publisher.start()
    logger.info("rlhf_worker_started", topic=active_settings.kafka_topic_rlhf_feedback)

    try:
        async for message in consumer:
            try:
                payload = RLHFFeedbackPayload.model_validate(message.value)
                append_feedback_record(active_settings.rlhf_feedback_store_path, payload)
                pending_since_last_train += 1

                total_samples = count_feedback_records(active_settings.rlhf_feedback_store_path)
                should_train = (
                    total_samples >= active_settings.rlhf_training_min_samples
                    and pending_since_last_train >= active_settings.rlhf_training_batch_size
                )

                if should_train:
                    records = load_feedback_records(active_settings.rlhf_feedback_store_path)
                    model_payload = train_reward_model(records)
                    save_reward_model(model_payload, active_settings.rlhf_reward_model_path)
                    pending_since_last_train = 0

                    await publisher.publish_event(
                        active_settings.kafka_topic_rlhf_model_updates,
                        {
                            "event_type": "RLHFRewardModelUpdated",
                            "sample_count": model_payload["sample_count"],
                            "positive_ratio": round(float(model_payload["positive_ratio"]), 4),
                            "reward_model_path": active_settings.rlhf_reward_model_path,
                        },
                    )
                    logger.info(
                        "rlhf_model_retrained",
                        sample_count=model_payload["sample_count"],
                        positive_ratio=round(float(model_payload["positive_ratio"]), 4),
                    )

                await consumer.commit()
            except Exception as exc:
                logger.exception("rlhf_worker_message_failed", error=str(exc))
                await consumer.commit()
    finally:
        await consumer.stop()
        await publisher.stop()
        logger.info("rlhf_worker_stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
