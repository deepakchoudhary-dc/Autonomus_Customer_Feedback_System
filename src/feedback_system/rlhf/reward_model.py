from __future__ import annotations

import json
import math
import re
from pathlib import Path

from feedback_system.rlhf.schemas import RLHFFeedbackPayload

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _record_text(record: RLHFFeedbackPayload) -> str:
    context = " ".join(record.retrieved_context)
    return f"{record.prompt} {context} {record.response}"


def train_reward_model(records: list[RLHFFeedbackPayload]) -> dict[str, object]:
    if len(records) < 2:
        raise ValueError("At least 2 feedback records are required")

    positive_count = 0
    negative_count = 0
    positive_tokens: dict[str, int] = {}
    negative_tokens: dict[str, int] = {}

    for record in records:
        tokens = _tokenize(_record_text(record))
        if record.rating > 0:
            positive_count += 1
            for token in tokens:
                positive_tokens[token] = positive_tokens.get(token, 0) + 1
        else:
            negative_count += 1
            for token in tokens:
                negative_tokens[token] = negative_tokens.get(token, 0) + 1

    vocabulary = set(positive_tokens) | set(negative_tokens)
    token_weights: dict[str, float] = {}
    for token in vocabulary:
        pos = positive_tokens.get(token, 0)
        neg = negative_tokens.get(token, 0)
        token_weights[token] = math.log((pos + 1.0) / (neg + 1.0))

    bias = math.log((positive_count + 1.0) / (negative_count + 1.0))
    positive_ratio = positive_count / max(1, len(records))

    return {
        "token_weights": token_weights,
        "bias": bias,
        "sample_count": len(records),
        "positive_ratio": positive_ratio,
    }


def score_text(model_payload: dict[str, object], text: str) -> float:
    token_weights = model_payload.get("token_weights", {})
    if not isinstance(token_weights, dict):
        return 0.5

    score = float(model_payload.get("bias", 0.0))
    for token in _tokenize(text):
        score += float(token_weights.get(token, 0.0))

    return 1.0 / (1.0 + math.exp(-score))


def save_reward_model(model_payload: dict[str, object], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(model_payload, handle, indent=2)


def load_reward_model(model_path: str) -> dict[str, object] | None:
    path = Path(model_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
