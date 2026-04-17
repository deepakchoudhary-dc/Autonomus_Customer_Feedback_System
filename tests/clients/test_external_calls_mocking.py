from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from feedback_system.config import Settings


@pytest.mark.asyncio
async def test_embedding_client_uses_mocked_openai_call() -> None:
    with patch("feedback_system.clients.embedding.AsyncOpenAI") as mock_openai_class:
        mock_client = Mock()
        mock_client.embeddings.create = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
        )
        mock_openai_class.return_value = mock_client

        from feedback_system.clients.embedding import EmbeddingClient

        client = EmbeddingClient(Settings(openai_api_key="test-key"))
        result = await client.create_embedding("login fails")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_vector_store_upsert_uses_mocked_pinecone_and_to_thread() -> None:
    with patch("feedback_system.clients.vector_store.Pinecone") as mock_pinecone, patch(
        "feedback_system.clients.vector_store.asyncio.to_thread", new_callable=AsyncMock
    ) as mock_to_thread:
        mock_index = Mock()
        mock_pinecone.return_value.Index.return_value = mock_index

        from feedback_system.clients.vector_store import VectorStoreClient

        client = VectorStoreClient(
            Settings(
                pinecone_api_key="test-key",
                pinecone_index_name="test-index",
                pinecone_namespace="test-ns",
            )
        )
        await client.upsert_feedback_vector(
            ticket_id="T-1",
            source_platform="zendesk",
            customer_email="user@example.com",
            embedding=[0.1, 0.2],
        )

        mock_to_thread.assert_awaited_once()
        called_kwargs = mock_to_thread.await_args.kwargs
        assert called_kwargs["namespace"] == "test-ns"


@pytest.mark.asyncio
async def test_create_jira_epic_uses_mocked_httpx_client() -> None:
    from feedback_system.integrations.jira_sync import CriticalBugEvent, create_jira_epic

    event = CriticalBugEvent(
        ai_summary="Login failures on Android",
        root_cause_hypothesis="Token refresh regression",
        affected_customer_ids=["cust-1"],
    )
    settings = Settings(jira_project_key="FB", jira_issue_type_epic="Epic")

    response = Mock(status_code=201)
    response.json.return_value = {"key": "FB-100"}
    response.raise_for_status = Mock()

    jira_client = Mock()
    jira_client.post = AsyncMock(return_value=response)

    created = await create_jira_epic(jira_client, event, settings)

    assert created.issue_key == "FB-100"
    jira_client.post.assert_awaited_once()
