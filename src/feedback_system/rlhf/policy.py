from __future__ import annotations

from feedback_system.rlhf.reward_model import load_reward_model, score_text


class RewardPolicy:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path

    def _load(self) -> dict[str, object] | None:
        return load_reward_model(self._model_path)

    def select_best_response(
        self,
        *,
        prompt: str,
        retrieved_context: list[str],
        candidates: list[str],
    ) -> str:
        if not candidates:
            return ""

        reward_model = self._load()
        if reward_model is None:
            return candidates[0]

        context = " ".join(retrieved_context)
        scored = []
        for candidate in candidates:
            text = f"{prompt} {context} {candidate}"
            scored.append((score_text(reward_model, text), candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
