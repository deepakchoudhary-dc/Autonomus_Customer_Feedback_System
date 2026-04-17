# Autonomous Customer Feedback System

This repository implements an event-driven, agentic customer feedback platform with the following closed-loop flow:

1. Ingest customer feedback via FastAPI webhook.
2. Generate embeddings and upsert vectors to Pinecone.
3. Publish feedback events to Kafka.
4. Detect repeated issue spikes and emit critical bug events.
5. Create Jira epics for critical anomalies.
6. Draft grounded resolutions using a self-reflective LangGraph pipeline.
7. Publish customer resolution notifications when Jira issues are marked resolved.
8. Predict churn risk continuously from live feedback streams.
9. Automate RLHF reward model updates from reviewer preference feedback.

## Advanced Capability Coverage

- Multimodal ingestion endpoint accepts text, image URLs, audio transcript, and video transcript signals.
- Predictive churn pipeline supports API scoring and streaming churn alerts.
- RLHF loop supports feedback capture, reward model training, model update events, and policy-guided response ranking.

## Core Stack

- Python 3.11+
- FastAPI (async APIs)
- SQLAlchemy 2.0 (async ORM)
- Alembic migrations
- Apache Kafka (event bus)
- Pinecone (vector store)
- LangChain + LangGraph (agentic orchestration)
- Pydantic v2 (strict schemas)
- structlog (JSON logs)
- Pytest (unit test coverage with mocked integrations)

## Runbook

1. Install dependencies:

```bash
make install
```

2. Start local infrastructure:

```bash
docker compose up -d
```

3. Apply DB migrations:

```bash
make migrate
```

4. Start API server:

```bash
make run
```

5. Start workers in separate terminals:

```bash
make worker-anomaly
make worker-jira-sync
make worker-resolution-notifier
make worker-churn
make worker-rlhf
```

6. Run tests:

```bash
make test
```

7. Train churn model artifact (optional, if you have labeled churn data):

```bash
make train-churn-model
```

8. Run one RLHF retraining cycle manually (optional):

```bash
make run-rlhf-cycle
```

9. Evaluate RAG quality (optional):

```bash
make evaluate-rag
```

## API Surface

- POST /api/v1/webhooks/feedback
- POST /api/v1/webhooks/feedback-multimodal
- POST /api/v1/resolution/draft
- POST /api/v1/churn/predict
- POST /api/v1/rlhf/feedback
- POST /api/v1/rlhf/train

## Privacy and Git Hygiene

- Runtime secrets are loaded from .env and excluded from version control.
- Derived model artifacts and RLHF raw feedback stores are ignored in .gitignore.
- Use .env.example as the only committed environment configuration reference.
