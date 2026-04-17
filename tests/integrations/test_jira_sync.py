from feedback_system.config import Settings
from feedback_system.integrations.jira_sync import (
    CriticalBugEvent,
    _build_adf_description,
    _build_jira_payload,
    _derive_cluster_key,
)


def test_derive_cluster_key_is_stable() -> None:
    event = CriticalBugEvent(
        ai_summary="Android login failures",
        root_cause_hypothesis="Token refresh bug on older Android OS versions",
        affected_customer_ids=["cust-1", "cust-2"],
    )

    assert _derive_cluster_key(event) == _derive_cluster_key(event)


def test_build_adf_description_includes_customer_links() -> None:
    event = CriticalBugEvent(
        ai_summary="Checkout crash",
        root_cause_hypothesis="Frontend regression from release 2.3.1",
        affected_customer_ids=["cust-100"],
    )

    adf = _build_adf_description(event, "https://internal.example.com/customers")
    assert adf["type"] == "doc"
    assert adf["content"][2]["type"] == "bulletList"
    link_mark = adf["content"][2]["content"][0]["content"][0]["content"][0]["marks"][0]
    assert link_mark["attrs"]["href"].endswith("/cust-100")


def test_build_jira_payload_maps_event_to_epic_fields() -> None:
    event = CriticalBugEvent(
        ai_summary="Billing mismatch",
        root_cause_hypothesis="Tax service timeout returns zero tax",
        affected_customer_ids=[],
    )
    settings = Settings(
        jira_project_key="FB",
        jira_issue_type_epic="Epic",
    )

    payload = _build_jira_payload(event, settings)

    assert payload["fields"]["project"]["key"] == "FB"
    assert payload["fields"]["summary"] == "Billing mismatch"
    assert payload["fields"]["issuetype"]["name"] == "Epic"
