from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Autonomous Customer Feedback System"
    log_level: str = "INFO"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_multimodal_model: str = "gpt-4o-mini"
    openai_resolution_model: str = "gpt-4o"
    openai_evaluator_model: str = "gpt-4o-mini"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "customer-feedback"
    pinecone_namespace: str = "feedback"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_feedback_events: str = "feedback-events"
    kafka_topic_feedback_multimodal_events: str = "feedback-multimodal-events"
    kafka_topic_critical_bugs: str = "critical-bugs-identified"
    kafka_topic_jira_resolved: str = "jira-issues-resolved"
    kafka_topic_customer_notifications: str = "customer-resolution-notifications"
    kafka_topic_churn_alerts: str = "customer-churn-alerts"
    kafka_topic_rlhf_feedback: str = "rlhf-feedback-events"
    kafka_topic_rlhf_model_updates: str = "rlhf-model-updates"
    kafka_consumer_group_anomaly: str = "anomaly-detector-worker"
    kafka_consumer_group_jira: str = "jira-sync-worker"
    kafka_consumer_group_resolution_notifier: str = "resolution-notifier-worker"
    kafka_consumer_group_churn_predictor: str = "churn-predictor-worker"
    kafka_consumer_group_rlhf: str = "rlhf-trainer-worker"
    anomaly_cluster_threshold: int = 3
    churn_alert_threshold: float = 0.7
    churn_model_path: str = "artifacts/churn_model.json"
    rlhf_feedback_store_path: str = "data/rlhf_feedback.jsonl"
    rlhf_reward_model_path: str = "artifacts/reward_model.json"
    rlhf_training_min_samples: int = 20
    rlhf_training_batch_size: int = 10
    database_url: str = "postgresql+asyncpg://feedback:feedback@localhost:5432/feedback_db"
    jira_base_url: str = "https://your-domain.atlassian.net"
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "FB"
    jira_issue_type_epic: str = "Epic"
    internal_dashboard_base_url: str = "https://internal.example.com/customers"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
