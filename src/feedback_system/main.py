from fastapi import FastAPI

from feedback_system.churn.api import router as churn_router
from feedback_system.churn.model import ChurnModel
from feedback_system.clients.embedding import EmbeddingClient
from feedback_system.clients.multimodal import MultimodalUnderstandingClient
from feedback_system.clients.vector_store import VectorStoreClient
from feedback_system.config import get_settings
from feedback_system.events.publisher import EventPublisher
from feedback_system.ingestion.api import router as ingestion_router
from feedback_system.logging import configure_logging
from feedback_system.resolution.api import router as resolution_router
from feedback_system.rlhf.api import router as rlhf_router


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0")

    app.state.settings = settings
    app.state.embedding_client = EmbeddingClient(settings)
    app.state.multimodal_understanding_client = MultimodalUnderstandingClient(settings)
    app.state.vector_store_client = VectorStoreClient(settings)
    app.state.event_publisher = EventPublisher(settings)
    app.state.churn_model = ChurnModel(settings.churn_model_path)

    @app.on_event("startup")
    async def startup_event() -> None:
        await app.state.event_publisher.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await app.state.event_publisher.stop()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(ingestion_router)
    app.include_router(resolution_router)
    app.include_router(churn_router)
    app.include_router(rlhf_router)
    return app


app = create_app()
