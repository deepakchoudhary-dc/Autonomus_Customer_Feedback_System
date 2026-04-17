from __future__ import annotations

import json
from pathlib import Path

from feedback_system.rlhf.schemas import RLHFFeedbackPayload


def append_feedback_record(file_path: str, payload: RLHFFeedbackPayload) -> int:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload.model_dump(mode="json")) + "\n")

    return count_feedback_records(file_path)


def load_feedback_records(file_path: str) -> list[RLHFFeedbackPayload]:
    path = Path(file_path)
    if not path.exists():
        return []

    records: list[RLHFFeedbackPayload] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = RLHFFeedbackPayload.model_validate(json.loads(line))
            records.append(payload)
    return records


def count_feedback_records(file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
