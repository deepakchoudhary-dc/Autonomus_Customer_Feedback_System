from feedback_system.rlhf.policy import RewardPolicy
from feedback_system.rlhf.reward_model import save_reward_model, train_reward_model
from feedback_system.rlhf.schemas import RLHFFeedbackPayload


def test_reward_model_training_and_policy_selection(tmp_path) -> None:
    records = [
        RLHFFeedbackPayload(
            prompt="How can I reset password?",
            response="Go to settings and click reset password.",
            retrieved_context=["Password reset is in settings"],
            rating=1,
            reviewer_id="qa-1",
        ),
        RLHFFeedbackPayload(
            prompt="How can I reset password?",
            response="We cannot help with that.",
            retrieved_context=["Password reset is in settings"],
            rating=-1,
            reviewer_id="qa-2",
        ),
    ]

    model_payload = train_reward_model(records)
    output_path = tmp_path / "reward_model.json"
    save_reward_model(model_payload, str(output_path))

    policy = RewardPolicy(str(output_path))
    winner = policy.select_best_response(
        prompt="How can I reset password?",
        retrieved_context=["Password reset is in settings"],
        candidates=[
            "Go to settings and click reset password.",
            "We cannot help with that.",
        ],
    )

    assert winner == "Go to settings and click reset password."
