from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FEATURE_NAMES = [
    "recent_negative_feedback_count",
    "avg_sentiment_score",
    "unresolved_ticket_count",
    "avg_first_response_minutes",
    "weekly_engagement_drop_ratio",
]
TARGET_COLUMN = "churned"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _load_rows(input_csv: Path) -> tuple[list[dict[str, float]], list[int]]:
    features: list[dict[str, float]] = []
    targets: list[int] = []

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features.append({name: float(row[name]) for name in FEATURE_NAMES})
            targets.append(int(row[TARGET_COLUMN]))

    return features, targets


def _normalize(features: list[dict[str, float]]) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for feature in FEATURE_NAMES:
        values = [row[feature] for row in features]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        std = math.sqrt(variance) or 1.0
        means[feature] = mean
        stds[feature] = std

    normalized: list[dict[str, float]] = []
    for row in features:
        normalized.append({
            feature: (row[feature] - means[feature]) / stds[feature]
            for feature in FEATURE_NAMES
        })

    return normalized, means, stds


def train_model(
    normalized_features: list[dict[str, float]],
    targets: list[int],
    epochs: int = 600,
    learning_rate: float = 0.05,
) -> tuple[dict[str, float], float]:
    weights = {feature: 0.0 for feature in FEATURE_NAMES}
    bias = 0.0

    sample_count = len(normalized_features)
    for _ in range(epochs):
        grad_w = {feature: 0.0 for feature in FEATURE_NAMES}
        grad_b = 0.0

        for row, target in zip(normalized_features, targets, strict=True):
            score = bias + sum(weights[feature] * row[feature] for feature in FEATURE_NAMES)
            prediction = _sigmoid(score)
            error = prediction - target

            for feature in FEATURE_NAMES:
                grad_w[feature] += error * row[feature]
            grad_b += error

        for feature in FEATURE_NAMES:
            weights[feature] -= learning_rate * (grad_w[feature] / sample_count)
        bias -= learning_rate * (grad_b / sample_count)

    return weights, bias


def evaluate(
    normalized_features: list[dict[str, float]],
    targets: list[int],
    weights: dict[str, float],
    bias: float,
) -> float:
    correct = 0
    for row, target in zip(normalized_features, targets, strict=True):
        score = bias + sum(weights[feature] * row[feature] for feature in FEATURE_NAMES)
        prediction = 1 if _sigmoid(score) >= 0.5 else 0
        if prediction == target:
            correct += 1

    return correct / max(1, len(targets))


def save_model(
    output_path: Path,
    *,
    weights: dict[str, float],
    bias: float,
    means: dict[str, float],
    stds: dict[str, float],
    accuracy: float,
    sample_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "bias": bias,
        "normalization": {
            "means": means,
            "stds": stds,
        },
        "metrics": {
            "training_accuracy": accuracy,
            "sample_count": sample_count,
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--input", required=True, type=Path, help="CSV file with churn labels")
    parser.add_argument("--output", required=True, type=Path, help="Output model JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, targets = _load_rows(args.input)
    if len(features) < 10:
        raise ValueError("At least 10 labeled samples are required to train churn model")

    normalized, means, stds = _normalize(features)
    weights, bias = train_model(normalized, targets)
    accuracy = evaluate(normalized, targets, weights, bias)
    save_model(
        args.output,
        weights=weights,
        bias=bias,
        means=means,
        stds=stds,
        accuracy=accuracy,
        sample_count=len(features),
    )


if __name__ == "__main__":
    main()
