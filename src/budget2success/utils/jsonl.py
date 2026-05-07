from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from budget2success._pydantic_compat import BaseModel


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if isinstance(record, BaseModel):
            payload = record.model_dump(mode="json")
        else:
            payload = record
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
