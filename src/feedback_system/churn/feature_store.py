from __future__ import annotations

import re
from dataclasses import dataclass

from feedback_system.churn.schemas import CustomerChurnFeatures, FeedbackSignalEvent


_NEGATIVE_TOKENS = {
    "broken",
    "bug",
    "fail",
    "failed",
    "error",
    "crash",
    "refund",
    "angry",
    "frustrated",
    "issue",
    "problem",
}
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class CustomerAggregate:
    negative_feedback_count: int = 0
    sentiment_sum: float = 0.0
    sentiment_samples: int = 0
    unresolved_ticket_count: int = 0
    avg_first_response_minutes: float = 45.0
    weekly_engagement_drop_ratio: float = 0.2


class CustomerFeatureStore:
    def __init__(self) -> None:
        self._store: dict[str, CustomerAggregate] = {}

    def _sentiment_score(self, raw_content: str) -> float:
        tokens = _WORD_PATTERN.findall(raw_content.lower())
        if not tokens:
            return 0.0

        negative_hits = sum(1 for token in tokens if token in _NEGATIVE_TOKENS)
        score = -(negative_hits / max(1, len(tokens) / 4))
        return max(-1.0, min(1.0, score))

    def _customer_id(self, customer_email: str) -> str:
        return customer_email.split("@", maxsplit=1)[0]

    def update(self, event: FeedbackSignalEvent) -> CustomerChurnFeatures:
        customer_id = self._customer_id(event.customer_email)
        aggregate = self._store.setdefault(customer_id, CustomerAggregate())

        sentiment = self._sentiment_score(event.raw_content)
        aggregate.sentiment_sum += sentiment
        aggregate.sentiment_samples += 1

        if sentiment < -0.1:
            aggregate.negative_feedback_count += 1
            aggregate.unresolved_ticket_count += 1
            aggregate.weekly_engagement_drop_ratio = min(
                1.0,
                aggregate.weekly_engagement_drop_ratio + 0.05,
            )
        else:
            aggregate.unresolved_ticket_count = max(0, aggregate.unresolved_ticket_count - 1)
            aggregate.weekly_engagement_drop_ratio = max(
                0.0,
                aggregate.weekly_engagement_drop_ratio - 0.02,
            )

        return CustomerChurnFeatures(
            customer_id=customer_id,
            recent_negative_feedback_count=aggregate.negative_feedback_count,
            avg_sentiment_score=(aggregate.sentiment_sum / max(1, aggregate.sentiment_samples)),
            unresolved_ticket_count=aggregate.unresolved_ticket_count,
            avg_first_response_minutes=aggregate.avg_first_response_minutes,
            weekly_engagement_drop_ratio=aggregate.weekly_engagement_drop_ratio,
        )
