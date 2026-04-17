from feedback_system.churn.model import ChurnModel
from feedback_system.churn.schemas import CustomerChurnFeatures


def test_churn_model_predicts_probability_in_range(tmp_path) -> None:
    model = ChurnModel(str(tmp_path / "churn_model.json"))
    payload = CustomerChurnFeatures(
        customer_id="cust-123",
        recent_negative_feedback_count=4,
        avg_sentiment_score=-0.7,
        unresolved_ticket_count=3,
        avg_first_response_minutes=80.0,
        weekly_engagement_drop_ratio=0.45,
    )

    probability = model.predict_probability(payload)
    assert 0.0 <= probability <= 1.0
    assert model.risk_level(probability) in {"low", "medium", "high"}
