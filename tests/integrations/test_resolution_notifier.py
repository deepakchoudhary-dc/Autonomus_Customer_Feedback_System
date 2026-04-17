from feedback_system.integrations.resolution_notifier import (
    JiraIssueResolvedEvent,
    build_customer_notification,
)


def test_build_customer_notification_contains_resolution_payload() -> None:
    event = JiraIssueResolvedEvent(
        issue_key="FB-123",
        resolution_summary="Fixed token refresh issue in Android auth flow",
        affected_customer_ids=["cust-1", "cust-2"],
    )

    payload = build_customer_notification(event, "cust-1")

    assert payload["event_type"] == "CustomerIssueResolved"
    assert payload["customer_id"] == "cust-1"
    assert payload["issue_key"] == "FB-123"
    assert "Fixed token refresh issue" in payload["message"]
