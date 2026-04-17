import asyncio

try:
    from pinecone import Pinecone
except ModuleNotFoundError:
    Pinecone = None

from feedback_system.config import Settings


class VectorStoreClient:
    def __init__(self, settings: Settings) -> None:
        if Pinecone is None:
            raise RuntimeError("pinecone package is not installed. Run `make install` first.")

        self._index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
        self._namespace = settings.pinecone_namespace

    async def upsert_feedback_vector(
        self,
        *,
        ticket_id: str,
        source_platform: str,
        customer_email: str,
        embedding: list[float],
        vector_id: str | None = None,
        extra_metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        metadata = {
            "ticket_id": ticket_id,
            "source_platform": source_platform,
            "customer_email": customer_email,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        vector_payload = {
            "id": vector_id or ticket_id,
            "values": embedding,
            "metadata": metadata,
        }

        # Pinecone's SDK is sync-only for this operation, so run it off the event loop.
        await asyncio.to_thread(
            self._index.upsert,
            vectors=[vector_payload],
            namespace=self._namespace,
        )
