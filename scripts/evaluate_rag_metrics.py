from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def _parse_contexts(raw_value: str) -> list[str]:
    value = raw_value.strip()
    if not value:
        return []

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass

    return [chunk.strip() for chunk in value.split("||") if chunk.strip()]


def _build_dataset(df: pd.DataFrame):
    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets package is required for RAG evaluation") from exc

    return Dataset.from_dict(
        {
            "question": df["question"].astype(str).tolist(),
            "answer": df["answer"].astype(str).tolist(),
            "contexts": [_parse_contexts(str(value)) for value in df["contexts"].tolist()],
            "ground_truth": df["ground_truth"].astype(str).tolist(),
        }
    )


def evaluate_rag_metrics(input_csv: Path, output_json: Path) -> dict[str, float]:
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ragas is not installed. Install optional extra with: uv sync --extra eval"
        ) from exc

    df = pd.read_csv(input_csv)
    required_columns = {"question", "answer", "contexts", "ground_truth"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        msg = f"CSV is missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    dataset = _build_dataset(df)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    detailed_results = result.to_pandas()
    summary_scores = {
        "faithfulness": float(detailed_results["faithfulness"].mean()),
        "answer_relevancy": float(detailed_results["answer_relevancy"].mean()),
        "context_precision": float(detailed_results["context_precision"].mean()),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    detailed_results.to_json(output_json, orient="records", indent=2)

    # To integrate with Arize Phoenix, emit summary_scores and the raw row-level
    # traces as OpenInference spans alongside prompt/context metadata.
    logger.info("ragas_summary_scores", **summary_scores)
    return summary_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality metrics with Ragas")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_rag_metrics(args.input, args.output)


if __name__ == "__main__":
    main()
