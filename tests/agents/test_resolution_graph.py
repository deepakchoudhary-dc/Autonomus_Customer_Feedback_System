import pytest

from feedback_system.agents.resolution_graph import ResolutionDependencies, build_resolution_graph


class FakeResolutionRuntime:
    def __init__(self, hallucination_sequence: list[bool]) -> None:
        self.hallucination_sequence = hallucination_sequence
        self.retrieve_calls = 0
        self.generate_calls = 0
        self.embed_calls = 0

    async def embed_query(self, query: str) -> list[float]:
        self.embed_calls += 1
        return [0.11, 0.22, 0.33]

    async def retrieve_chunks(self, embedding: list[float], top_k: int) -> list[str]:
        self.retrieve_calls += 1
        return [f"retrieved-context-{self.retrieve_calls}"]

    async def generate_response(self, customer_query: str, chunks: list[str]) -> str:
        self.generate_calls += 1
        return f"draft-{self.generate_calls}-for-{customer_query}"

    async def evaluate_hallucination(self, draft_response: str, chunks: list[str]) -> bool:
        if not self.hallucination_sequence:
            return False
        return self.hallucination_sequence.pop(0)


@pytest.mark.asyncio
async def test_resolution_graph_retries_when_hallucination_detected() -> None:
    runtime = FakeResolutionRuntime(hallucination_sequence=[True, False])
    deps = ResolutionDependencies(
        embed_query=runtime.embed_query,
        retrieve_chunks=runtime.retrieve_chunks,
        generate_response=runtime.generate_response,
        evaluate_hallucination=runtime.evaluate_hallucination,
    )

    graph = build_resolution_graph(dependencies=deps)
    result = await graph.ainvoke(
        {
            "customer_query": "My refund has not arrived",
            "retrieved_chunks": [],
            "draft_response": "",
            "hallucination_score": False,
            "retry_count": 0,
        }
    )

    assert runtime.retrieve_calls == 2
    assert runtime.generate_calls == 2
    assert result["retry_count"] == 1
    assert result["hallucination_score"] is False


@pytest.mark.asyncio
async def test_resolution_graph_escalates_after_three_hallucinations() -> None:
    runtime = FakeResolutionRuntime(hallucination_sequence=[True, True, True])
    deps = ResolutionDependencies(
        embed_query=runtime.embed_query,
        retrieve_chunks=runtime.retrieve_chunks,
        generate_response=runtime.generate_response,
        evaluate_hallucination=runtime.evaluate_hallucination,
    )

    graph = build_resolution_graph(dependencies=deps)
    result = await graph.ainvoke(
        {
            "customer_query": "Unable to reset password",
            "retrieved_chunks": [],
            "draft_response": "",
            "hallucination_score": False,
            "retry_count": 0,
        }
    )

    assert runtime.retrieve_calls == 3
    assert result["retry_count"] == 3
    assert result["hallucination_score"] is True
    assert "Escalated to a human support specialist" in result["draft_response"]
