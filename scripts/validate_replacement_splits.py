#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


FIELDS = ["split", "check", "status", "observed", "expected", "notes"]


def validate_replacement_splits(
    *,
    canitedit: str | Path,
    bigcodebench: str | Path,
    output: str | Path,
    min_canitedit_tasks: int = 100,
    min_canitedit_test_coverage: float = 0.80,
    expected_bigcodebench_tasks: int = 148,
) -> tuple[Path, bool]:
    rows: list[dict[str, Any]] = []
    canitedit_tasks = _load_tasks(Path(canitedit), split_name="canitedit", rows=rows)
    bigcodebench_tasks = _load_tasks(Path(bigcodebench), split_name="bigcodebench_hard", rows=rows)

    _check(
        rows,
        "canitedit",
        "task_count",
        len(canitedit_tasks) >= min_canitedit_tasks,
        len(canitedit_tasks),
        f">={min_canitedit_tasks}",
        "real CanItEdit split size",
    )
    test_count = sum(1 for task in canitedit_tasks if str((task.metadata or {}).get("tests") or "").strip())
    coverage = test_count / len(canitedit_tasks) if canitedit_tasks else 0.0
    _check(
        rows,
        "canitedit",
        "provided_test_coverage",
        coverage >= min_canitedit_test_coverage,
        f"{coverage:.6f}",
        f">={min_canitedit_test_coverage:.2f}",
        f"{test_count}/{len(canitedit_tasks)} rows have nonempty metadata.tests",
    )
    _check(
        rows,
        "canitedit",
        "not_toy_smoke",
        len({task.task_id for task in canitedit_tasks}) >= min_canitedit_tasks,
        len({task.task_id for task in canitedit_tasks}),
        f">={min_canitedit_tasks}",
        "unique task IDs",
    )

    _check(
        rows,
        "bigcodebench_hard",
        "task_count",
        len(bigcodebench_tasks) == expected_bigcodebench_tasks,
        len(bigcodebench_tasks),
        expected_bigcodebench_tasks,
        "expected BigCodeBench-Hard task count",
    )

    for split_name, tasks in (("canitedit", canitedit_tasks), ("bigcodebench_hard", bigcodebench_tasks)):
        _check(
            rows,
            split_name,
            "chat_completion_compatible",
            all(_truthy((task.metadata or {}).get("chat_completion_compatible")) for task in tasks),
            _count_truthy(tasks, "chat_completion_compatible"),
            len(tasks),
            "every row must be chat-completion compatible",
        )
        _check(
            rows,
            split_name,
            "requires_docker_false",
            all(not _truthy((task.metadata or {}).get("requires_docker")) for task in tasks),
            sum(1 for task in tasks if not _truthy((task.metadata or {}).get("requires_docker"))),
            len(tasks),
            "local verification path must not require Docker",
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    ok = all(row["status"] == "pass" for row in rows)
    return output_path, ok


def _load_tasks(path: Path, *, split_name: str, rows: list[dict[str, Any]]) -> list[TaskRecord]:
    if not path.exists():
        _check(rows, split_name, "file_exists", False, "missing", str(path), "split file missing")
        return []
    try:
        tasks = [TaskRecord.model_validate(row) for row in read_jsonl(path)]
    except Exception as exc:
        _check(rows, split_name, "schema_valid", False, type(exc).__name__, "valid TaskRecord JSONL", str(exc))
        return []
    _check(rows, split_name, "file_exists", True, str(path), str(path), "split file present")
    _check(rows, split_name, "schema_valid", True, len(tasks), "valid TaskRecord JSONL", "all rows validate")
    return tasks


def _check(
    rows: list[dict[str, Any]],
    split: str,
    check: str,
    ok: bool,
    observed: Any,
    expected: Any,
    notes: str,
) -> None:
    rows.append(
        {
            "split": split,
            "check": check,
            "status": "pass" if ok else "fail",
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def _count_truthy(tasks: list[TaskRecord], key: str) -> int:
    return sum(1 for task in tasks if _truthy((task.metadata or {}).get(key)))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate chat-completion-compatible replacement task splits.")
    parser.add_argument("--canitedit", required=True)
    parser.add_argument("--bigcodebench", required=True)
    parser.add_argument("--output", default="reports/tables/replacement_split_validation.csv")
    parser.add_argument("--min-canitedit-tasks", type=int, default=100)
    parser.add_argument("--min-canitedit-test-coverage", type=float, default=0.80)
    parser.add_argument("--expected-bigcodebench-tasks", type=int, default=148)
    args = parser.parse_args()
    output_path, ok = validate_replacement_splits(
        canitedit=args.canitedit,
        bigcodebench=args.bigcodebench,
        output=args.output,
        min_canitedit_tasks=args.min_canitedit_tasks,
        min_canitedit_test_coverage=args.min_canitedit_test_coverage,
        expected_bigcodebench_tasks=args.expected_bigcodebench_tasks,
    )
    print(output_path)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
