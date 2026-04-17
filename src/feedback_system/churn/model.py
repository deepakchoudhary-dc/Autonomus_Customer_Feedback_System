from __future__ import annotations

import json
import math
from pathlib import Path

from feedback_system.churn.schemas import CustomerChurnFeatures


FEATURE_NAMES = [
    "recent_negative_feedback_count",
    "avg_sentiment_score",
    "unresolved_ticket_count",
    "avg_first_response_minutes",
    "weekly_engagement_drop_ratio",
]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class ChurnModel:
    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._weights: dict[str, float] = {
            "recent_negative_feedback_count": 0.9,
            "avg_sentiment_score": -1.4,
            "unresolved_ticket_count": 0.8,
            "avg_first_response_minutes": 0.01,
            "weekly_engagement_drop_ratio": 2.0,
        }
        self._bias: float = -2.2
        self._means: dict[str, float] = {feature: 0.0 for feature in FEATURE_NAMES}
        self._stds: dict[str, float] = {feature: 1.0 for feature in FEATURE_NAMES}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self._model_path.exists():
            return

        with self._model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        weights = payload.get("weights") or {}
        means = payload.get("normalization", {}).get("means") or {}
        stds = payload.get("normalization", {}).get("stds") or {}
        for feature in FEATURE_NAMES:
            if feature in weights:
                self._weights[feature] = float(weights[feature])
            if feature in means:
                self._means[feature] = float(means[feature])
            if feature in stds and float(stds[feature]) != 0.0:
                self._stds[feature] = float(stds[feature])

        self._bias = float(payload.get("bias", self._bias))

    def _normalized_feature(self, name: str, value: float) -> float:
        std = self._stds.get(name, 1.0)
        mean = self._means.get(name, 0.0)
        if std == 0.0:
            return value - mean
        return (value - mean) / std

    def predict_probability(self, payload: CustomerChurnFeatures) -> float:
        score = self._bias
        values = payload.model_dump()
        for feature_name in FEATURE_NAMES:
            normalized = self._normalized_feature(feature_name, float(values[feature_name]))
            score += normalized * self._weights[feature_name]
        return max(0.0, min(1.0, _sigmoid(score)))

    def risk_level(self, probability: float) -> str:
        if probability >= 0.8:
            return "high"
        if probability >= 0.5:
            return "medium"
        return "low"
