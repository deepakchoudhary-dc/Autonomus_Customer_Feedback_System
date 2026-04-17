from __future__ import annotations

import argparse
from pathlib import Path

from feedback_system.rlhf.reward_model import save_reward_model, train_reward_model
from feedback_system.rlhf.store import load_feedback_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RLHF training cycle")
    parser.add_argument("--feedback-store", required=True, type=Path, help="Input feedback jsonl path")
    parser.add_argument("--output-model", required=True, type=Path, help="Output reward model path")
    parser.add_argument(
        "--min-samples",
        required=False,
        default=20,
        type=int,
        help="Minimum samples required for training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_feedback_records(str(args.feedback_store))
    if len(records) < args.min_samples:
        raise ValueError(f"Need at least {args.min_samples} samples. Found {len(records)}")

    model_payload = train_reward_model(records)
    save_reward_model(model_payload, str(args.output_model))


if __name__ == "__main__":
    main()
