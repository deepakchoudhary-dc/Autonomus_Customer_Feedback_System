from __future__ import annotations

import re
from dataclasses import dataclass, field

from feedback_system.anomaly.schemas import CriticalBugDetectedEvent, FeedbackIngestedEvent


_WORD_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class ClusterAggregate:
    representative_text: str
    count: int = 0
    last_emitted_count: int = 0
    customer_ids: set[str] = field(default_factory=set)
    ticket_ids: list[str] = field(default_factory=list)


class FeedbackClusterDetector:
    def __init__(self, threshold: int = 3) -> None:
        if threshold < 2:
            msg = "threshold must be >= 2"
            raise ValueError(msg)

        self._threshold = threshold
        self._clusters: dict[str, ClusterAggregate] = {}

    def _cluster_key(self, raw_content: str) -> str:
        normalized = raw_content.lower()
        tokens = _WORD_PATTERN.findall(normalized)
        if not tokens:
            return "unknown-cluster"

        return "-".join(tokens[:8])

    def _build_summary(self, representative_text: str) -> str:
        snippet = representative_text.strip().replace("\n", " ")
        if len(snippet) > 110:
            snippet = snippet[:107].rstrip() + "..."
        return f"Customer issue spike: {snippet}"

    def _build_root_cause_hypothesis(self, representative_text: str, count: int) -> str:
        return (
            "A repeated complaint pattern was detected across multiple customers. "
            f"Observed {count} related reports. "
            "Likely root cause is a shared product-path regression linked to: "
            f"{representative_text.strip()}"
        )

    def ingest_feedback(self, event: FeedbackIngestedEvent) -> CriticalBugDetectedEvent | None:
        cluster_key = self._cluster_key(event.raw_content)
        aggregate = self._clusters.setdefault(
            cluster_key,
            ClusterAggregate(representative_text=event.raw_content),
        )

        aggregate.count += 1
        aggregate.customer_ids.add(event.customer_email)
        aggregate.ticket_ids.append(event.ticket_id)

        if (aggregate.count - aggregate.last_emitted_count) < self._threshold:
            return None

        aggregate.last_emitted_count = aggregate.count
        return CriticalBugDetectedEvent(
            ai_summary=self._build_summary(aggregate.representative_text),
            root_cause_hypothesis=self._build_root_cause_hypothesis(
                aggregate.representative_text,
                aggregate.count,
            ),
            affected_customer_ids=sorted(aggregate.customer_ids),
            cluster_key=cluster_key,
            trigger_count=aggregate.count,
            sample_ticket_ids=aggregate.ticket_ids[-5:],
        )
