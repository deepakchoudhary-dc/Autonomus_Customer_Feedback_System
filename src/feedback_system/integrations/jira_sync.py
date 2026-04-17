from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from aiokafka import AIOKafkaConsumer
except ModuleNotFoundError:
    AIOKafkaConsumer = None

from feedback_system.config import Settings, get_settings
from feedback_system.db.models import Anomaly

logger = structlog.get_logger(__name__)


class JiraRateLimitError(Exception):
    """Raised when Jira returns HTTP 429."""


class CriticalBugEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ai_summary: str = Field(min_length=5)
    root_cause_hypothesis: str = Field(min_length=5)
    affected_customer_ids: list[str] = Field(default_factory=list)
    anomaly_id: int | None = None


@dataclass(slots=True)
class JiraCreatedIssue:
    issue_key: str


def _derive_cluster_key(event: CriticalBugEvent) -> str:
    digest = hashlib.sha256(
        f"{event.ai_summary}|{event.root_cause_hypothesis}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _build_adf_description(event: CriticalBugEvent, dashboard_base_url: str) -> dict[str, Any]:
    bullet_items: list[dict[str, Any]] = []
    for customer_id in event.affected_customer_ids:
        customer_url = f"{dashboard_base_url.rstrip('/')}/{customer_id}"
        bullet_items.append(
            {
                "type": "listItem",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Customer {customer_id}",
                                "marks": [
                                    {
                                        "type": "link",
                                        "attrs": {"href": customer_url},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    content: list[dict[str, Any]] = [
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": "AI Root Cause Hypothesis:"}],
        },
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": event.root_cause_hypothesis}],
        },
    ]

    if bullet_items:
        content.append({"type": "bulletList", "content": bullet_items})

    return {
        "type": "doc",
        "version": 1,
        "content": content,
    }


def _build_jira_payload(event: CriticalBugEvent, settings: Settings) -> dict[str, Any]:
    return {
        "fields": {
            "project": {"key": settings.jira_project_key},
            "summary": event.ai_summary,
            "description": _build_adf_description(event, settings.internal_dashboard_base_url),
            "issuetype": {"name": settings.jira_issue_type_epic},
        }
    }


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((JiraRateLimitError, httpx.RequestError)),
)
async def create_jira_epic(
    jira_client: httpx.AsyncClient,
    event: CriticalBugEvent,
    settings: Settings,
) -> JiraCreatedIssue:
    payload = _build_jira_payload(event, settings)
    response = await jira_client.post("/rest/api/3/issue", json=payload)

    if response.status_code == 429:
        raise JiraRateLimitError("Jira rate limit reached")

    response.raise_for_status()
    issue_key = response.json()["key"]
    return JiraCreatedIssue(issue_key=issue_key)


async def persist_jira_link(event: CriticalBugEvent, issue_key: str) -> None:
    from feedback_system.db.session import AsyncSessionFactory

    async with AsyncSessionFactory() as session:
        async with session.begin():
            anomaly: Anomaly | None = None
            if event.anomaly_id is not None:
                anomaly = await session.get(Anomaly, event.anomaly_id)

            if anomaly is None:
                cluster_key = _derive_cluster_key(event)
                result = await session.execute(
                    select(Anomaly).where(Anomaly.cluster_key == cluster_key)
                )
                anomaly = result.scalar_one_or_none()

            if anomaly is None:
                anomaly = Anomaly(
                    cluster_key=_derive_cluster_key(event),
                    ai_summary=event.ai_summary,
                    root_cause_hypothesis=event.root_cause_hypothesis,
                    severity="high",
                    jira_issue_key=issue_key,
                )
                session.add(anomaly)
                return

            anomaly.ai_summary = event.ai_summary
            anomaly.root_cause_hypothesis = event.root_cause_hypothesis
            anomaly.jira_issue_key = issue_key


def _kafka_value_deserializer(raw_value: bytes) -> dict[str, Any]:
    return json.loads(raw_value.decode("utf-8"))


async def run_worker(settings: Settings | None = None) -> None:
    if AIOKafkaConsumer is None:
        msg = "aiokafka package is not installed. Run `make install` first."
        raise RuntimeError(msg)

    active_settings = settings or get_settings()
    consumer = AIOKafkaConsumer(
        active_settings.kafka_topic_critical_bugs,
        bootstrap_servers=active_settings.kafka_bootstrap_servers,
        group_id=active_settings.kafka_consumer_group_jira,
        enable_auto_commit=False,
        value_deserializer=_kafka_value_deserializer,
    )

    await consumer.start()
    logger.info("jira_sync_worker_started", topic=active_settings.kafka_topic_critical_bugs)

    try:
        async with httpx.AsyncClient(
            base_url=active_settings.jira_base_url,
            auth=(active_settings.jira_email, active_settings.jira_api_token),
            timeout=30.0,
        ) as jira_client:
            async for message in consumer:
                try:
                    event = CriticalBugEvent.model_validate(message.value)
                    created_issue = await create_jira_epic(jira_client, event, active_settings)
                    await persist_jira_link(event, created_issue.issue_key)
                    await consumer.commit()
                    logger.info(
                        "jira_issue_created",
                        issue_key=created_issue.issue_key,
                        anomaly_id=event.anomaly_id,
                    )
                except Exception as exc:
                    logger.exception("jira_sync_worker_message_failed", error=str(exc))
                    await consumer.commit()
    finally:
        await consumer.stop()
        logger.info("jira_sync_worker_stopped")


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
