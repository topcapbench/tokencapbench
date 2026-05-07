from __future__ import annotations

import json
from pathlib import Path

from .task_schema import TaskRecord


def load_tasks_jsonl(path: str | Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
            records.append(TaskRecord.model_validate(data))
    return records


def write_tasks_jsonl(path: str | Path, tasks: list[TaskRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n")
