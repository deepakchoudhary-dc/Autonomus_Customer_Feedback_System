install:
	uv sync --dev

run:
	uv run uvicorn feedback_system.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

test:
	uv run pytest

migrate:
	uv run alembic upgrade head

worker-jira-sync:
	uv run python -m feedback_system.integrations.jira_sync

worker-anomaly:
	uv run python -m feedback_system.anomaly.worker

worker-resolution-notifier:
	uv run python -m feedback_system.integrations.resolution_notifier

worker-churn:
	uv run python -m feedback_system.churn.worker

worker-rlhf:
	uv run python -m feedback_system.rlhf.worker

train-churn-model:
	uv run python scripts/train_churn_model.py --input data/churn_training.csv --output artifacts/churn_model.json

run-rlhf-cycle:
	uv run python scripts/run_rlhf_cycle.py --feedback-store data/rlhf_feedback.jsonl --output-model artifacts/reward_model.json

evaluate-rag:
	uv run python scripts/evaluate_rag_metrics.py --input data/rag_eval_dataset.csv --output artifacts/rag_eval_results.json
