from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypedDict

from pydantic import BaseModel

from feedback_system.config import Settings, get_settings
from feedback_system.rlhf.policy import RewardPolicy

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    from langchain_community.chat_models import ChatOpenAI

from langgraph.graph import END, StateGraph

try:
    from pinecone import Pinecone
except ModuleNotFoundError:
    Pinecone = None


class AgentState(TypedDict):
    customer_query: str
    retrieved_chunks: list[str]
    draft_response: str
    hallucination_score: bool
    retry_count: int


class HallucinationEvaluation(BaseModel):
    is_hallucinated: bool


@dataclass(slots=True)
class ResolutionDependencies:
    embed_query: Callable[[str], Awaitable[list[float]]]
    retrieve_chunks: Callable[[list[float], int], Awaitable[list[str]]]
    generate_response: Callable[[str, list[str]], Awaitable[str]]
    evaluate_hallucination: Callable[[str, list[str]], Awaitable[bool]]


class ResolutionRuntime:
    def __init__(self, settings: Settings) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not installed. Run `make install` first.")

        if Pinecone is None:
            raise RuntimeError("pinecone package is not installed. Run `make install` first.")

        self._embedding_model = settings.openai_embedding_model
        self._reward_policy = RewardPolicy(settings.rlhf_reward_model_path)
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._vector_index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
        self._namespace = settings.pinecone_namespace

        self._generator = ChatOpenAI(
            model=settings.openai_resolution_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        self._evaluator = ChatOpenAI(
            model=settings.openai_evaluator_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        ).with_structured_output(HallucinationEvaluation)

    async def embed_query(self, query: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=self._embedding_model,
            input=query,
        )
        return list(response.data[0].embedding)

    async def retrieve_chunks(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        result = await asyncio.to_thread(
            self._vector_index.query,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
        )
        chunks: list[str] = []
        for match in result.matches:
            metadata = getattr(match, "metadata", None) or {}
            chunk_text = metadata.get("raw_content") or metadata.get("content") or ""
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    async def generate_response(self, customer_query: str, retrieved_chunks: list[str]) -> str:
        context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No context available."
        base_prompt = (
            "You are a support resolution assistant. "
            "Answer only with facts in the context. "
            "If context is missing, state you need escalation.\n\n"
            f"Customer query:\n{customer_query}\n\n"
            f"Context:\n{context}"
        )

        candidates: list[str] = []
        for idx in range(3):
            prompt = (
                f"{base_prompt}\n\n"
                f"Candidate variant {idx + 1}: keep the response concise, actionable, and policy-safe."
            )
            response = await self._generator.ainvoke(prompt)
            candidate_text = response.content if isinstance(response.content, str) else str(response.content)
            candidates.append(candidate_text)

        return self._reward_policy.select_best_response(
            prompt=customer_query,
            retrieved_context=retrieved_chunks,
            candidates=candidates,
        )

    async def evaluate_hallucination(self, draft_response: str, retrieved_chunks: list[str]) -> bool:
        context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No context available."
        prompt = (
            "Determine if every factual claim in the draft response is explicitly supported by context. "
            "Return is_hallucinated=true if any claim is unsupported.\n\n"
            f"Draft response:\n{draft_response}\n\n"
            f"Context:\n{context}"
        )
        evaluation = await self._evaluator.ainvoke(prompt)
        return evaluation.is_hallucinated


def default_dependencies(settings: Settings | None = None) -> ResolutionDependencies:
    active_settings = settings or get_settings()
    runtime = ResolutionRuntime(active_settings)
    return ResolutionDependencies(
        embed_query=runtime.embed_query,
        retrieve_chunks=runtime.retrieve_chunks,
        generate_response=runtime.generate_response,
        evaluate_hallucination=runtime.evaluate_hallucination,
    )


async def retrieve_historical_context(
    state: AgentState,
    dependencies: ResolutionDependencies,
) -> AgentState:
    query_embedding = await dependencies.embed_query(state["customer_query"])
    retrieved_chunks = await dependencies.retrieve_chunks(query_embedding, 5)
    return {**state, "retrieved_chunks": retrieved_chunks}


async def generate_resolution(state: AgentState, dependencies: ResolutionDependencies) -> AgentState:
    draft_response = await dependencies.generate_response(
        state["customer_query"],
        state["retrieved_chunks"],
    )
    return {**state, "draft_response": draft_response}


async def evaluate_for_hallucinations(
    state: AgentState,
    dependencies: ResolutionDependencies,
) -> AgentState:
    hallucinated = await dependencies.evaluate_hallucination(
        state["draft_response"],
        state["retrieved_chunks"],
    )
    retry_count = state["retry_count"] + 1 if hallucinated else state["retry_count"]
    return {**state, "hallucination_score": hallucinated, "retry_count": retry_count}


async def human_escalation(state: AgentState) -> AgentState:
    return {
        **state,
        "draft_response": "Escalated to a human support specialist due to unresolved grounding checks.",
    }


def _route_after_evaluation(state: AgentState) -> str:
    if state["hallucination_score"] and state["retry_count"] < 3:
        return "retrieve_historical_context"
    if state["hallucination_score"] and state["retry_count"] >= 3:
        return "human_escalation"
    return END


def build_resolution_graph(
    dependencies: ResolutionDependencies | None = None,
):
    deps = dependencies or default_dependencies()

    workflow = StateGraph(AgentState)

    async def retrieve_node(state: AgentState) -> AgentState:
        return await retrieve_historical_context(state, deps)

    async def generate_node(state: AgentState) -> AgentState:
        return await generate_resolution(state, deps)

    async def evaluate_node(state: AgentState) -> AgentState:
        return await evaluate_for_hallucinations(state, deps)

    workflow.add_node("retrieve_historical_context", retrieve_node)
    workflow.add_node("generate_resolution", generate_node)
    workflow.add_node("evaluate_for_hallucinations", evaluate_node)
    workflow.add_node("human_escalation", human_escalation)

    workflow.set_entry_point("retrieve_historical_context")
    workflow.add_edge("retrieve_historical_context", "generate_resolution")
    workflow.add_edge("generate_resolution", "evaluate_for_hallucinations")
    workflow.add_conditional_edges(
        "evaluate_for_hallucinations",
        _route_after_evaluation,
        {
            "retrieve_historical_context": "retrieve_historical_context",
            "human_escalation": "human_escalation",
            END: END,
        },
    )
    workflow.add_edge("human_escalation", END)
    return workflow.compile()
