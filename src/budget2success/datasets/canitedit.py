from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl

DEFAULT_BUDGET_GRID = [128, 256, 512, 1024, 2048]
DEFAULT_LIMIT = 105


class CanItEditAdapter(BenchmarkSourceAdapter):
    """Loader for the CanItEdit instructional Python editing benchmark."""

    source_name = "canitedit"

    @classmethod
    def available(cls) -> bool:
        try:
            import datasets  # noqa: F401

            return True
        except ImportError:
            return False

    def load_tasks(self) -> list[TaskRecord]:
        style = str(self.config.kwargs.get("instruction_style") or self.config.kwargs.get("style") or "descriptive")
        if style not in {"descriptive", "lazy"}:
            raise ValueError("CanItEdit instruction_style must be 'descriptive' or 'lazy'.")
        limit = self.config.limit or DEFAULT_LIMIT
        rows, source_version, split_name = self._load_rows()
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(rows):
            if len(tasks) >= limit:
                break
            instruction = _instruction(row, style)
            before_code = str(row.get("before") or "")
            tests = str(row.get("tests") or "")
            original_id = _task_id(row, idx)
            task = TaskRecord(
                task_id=f"canitedit_{style}_{_safe_id(original_id)}",
                track="code_editing",
                source="canitedit",
                source_version=source_version,
                external_id=original_id,
                prompt=_format_prompt(before_code, instruction),
                verifier="canitedit",
                budget_grid=self.config.budget_grid or DEFAULT_BUDGET_GRID,
                metadata={
                    "instruction_style": style,
                    "before_code": before_code,
                    "instruction": instruction,
                    "after_code": row.get("after"),
                    "tests": tests,
                    "taxonomy": row.get("taxonomy"),
                    "paper_role": "paper_extension_or_appendix",
                    "chat_completion_compatible": True,
                    "requires_docker": False,
                    "official_harness_status": "provided_tests_available" if tests else "hidden_tests_unavailable",
                    "raw_index": idx,
                },
                external_eval={
                    "harness": "canitedit",
                    "task_id": original_id,
                    "split": split_name,
                    "instruction_style": style,
                    "source_version": source_version,
                },
            )
            tasks.append(task)
        if not tasks:
            raise RuntimeError("CanItEdit loader found no tasks.")
        return tasks

    def _load_rows(self) -> tuple[list[dict[str, Any]], str, str]:
        local_path = self.config.kwargs.get("path") or self.config.kwargs.get("local_export")
        if local_path:
            path = Path(str(local_path))
            if not path.exists():
                raise RuntimeError(f"CanItEdit local export not found: {path}")
            return _read_records(path), str(path), self.config.split or "local"

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` to load CanItEdit from Hugging Face.") from exc

        dataset_name = self.config.kwargs.get("dataset") or self.config.kwargs.get("hf_dataset") or "nuprl/CanItEdit"
        split = self.config.split or self.config.kwargs.get("split") or "test"
        ds = load_dataset(str(dataset_name), split=str(split))
        return [dict(row) for row in ds], str(dataset_name), str(split)


def load_canitedit_tasks(
    instruction_style: str = "descriptive",
    limit: int | None = None,
) -> list[TaskRecord]:
    from budget2success.datasets.base import AdapterConfig

    adapter = CanItEditAdapter(
        AdapterConfig(
            name="canitedit",
            split="test",
            limit=limit,
            budget_grid=DEFAULT_BUDGET_GRID,
            kwargs={"instruction_style": instruction_style},
        )
    )
    return adapter.load_tasks()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("rows", "data", "tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [{**value, "id": key} for key, value in payload.items()]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise RuntimeError(f"No CanItEdit records found in {path}")


def _instruction(row: dict[str, Any], style: str) -> str:
    key = f"instruction_{style}"
    value = row.get(key) or row.get("instruction") or row.get(style)
    if value in {None, ""}:
        raise RuntimeError(f"CanItEdit row missing {key}.")
    return str(value)


def _task_id(row: dict[str, Any], idx: int) -> str:
    for key in ("full_name", "task_id", "id", "name"):
        value = row.get(key)
        if value not in {None, ""} and not isinstance(value, (dict, list)):
            return str(value)
    return str(idx)


def _format_prompt(before_code: str, instruction: str) -> str:
    return (
        "Original code:\n"
        f"{before_code.strip()}\n\n"
        "Edit instruction:\n"
        f"{instruction.strip()}\n\n"
        "Return only the complete edited Python program."
    ).strip()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"
