from feedback_system.anomaly.detector import FeedbackClusterDetector
from feedback_system.anomaly.schemas import FeedbackIngestedEvent


def build_event(ticket_id: str, customer: str, content: str) -> FeedbackIngestedEvent:
    return FeedbackIngestedEvent(
        ticket_id=ticket_id,
        customer_email=customer,
        source_platform="zendesk",
        raw_content=content,
        ingested_at="2026-04-17T00:00:00Z",
    )


def test_detector_emits_after_threshold() -> None:
    detector = FeedbackClusterDetector(threshold=3)
    content = "Login fails with token refresh error on Android 10"

    first = detector.ingest_feedback(build_event("T1", "u1@example.com", content))
    second = detector.ingest_feedback(build_event("T2", "u2@example.com", content))
    third = detector.ingest_feedback(build_event("T3", "u3@example.com", content))

    assert first is None
    assert second is None
    assert third is not None
    assert third.trigger_count == 3
    assert len(third.affected_customer_ids) == 3


def test_detector_re_emits_on_next_threshold_window() -> None:
    detector = FeedbackClusterDetector(threshold=3)
    content = "Checkout fails after coupon apply"

    for idx in range(1, 4):
        detector.ingest_feedback(build_event(f"T{idx}", f"u{idx}@example.com", content))

    fourth = detector.ingest_feedback(build_event("T4", "u4@example.com", content))
    fifth = detector.ingest_feedback(build_event("T5", "u5@example.com", content))
    sixth = detector.ingest_feedback(build_event("T6", "u6@example.com", content))

    assert fourth is None
    assert fifth is None
    assert sixth is not None
    assert sixth.trigger_count == 6


def test_detector_requires_threshold_of_at_least_two() -> None:
    try:
        FeedbackClusterDetector(threshold=1)
        assert False, "Expected ValueError for invalid threshold"
    except ValueError:
        assert True
