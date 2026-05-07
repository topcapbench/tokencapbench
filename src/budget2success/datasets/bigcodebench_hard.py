from __future__ import annotations

import importlib.metadata
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


DEFAULT_BUDGET_GRID = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_LIMIT = 148


class BigCodeBenchHardAdapter(BenchmarkSourceAdapter):
    """Loader for the BigCodeBench-Hard practical Python code track."""

    source_name = "bigcodebench_hard"

    @classmethod
    def available(cls) -> bool:
        try:
            import bigcodebench  # noqa: F401

            return True
        except ImportError:
            try:
                import datasets  # noqa: F401

                return True
            except ImportError:
                return False

    def load_tasks(self) -> list[TaskRecord]:
        limit = self.config.limit or DEFAULT_LIMIT
        rows, source_version, split_name, package_version = self._load_rows()
        hard_rows = [row for row in rows if _is_hard(row)]
        selected = hard_rows if hard_rows else rows
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(selected):
            if len(tasks) >= limit:
                break
            original_id = _task_id(row, idx)
            prompt = _format_prompt(row)
            task = TaskRecord(
                task_id=f"bigcodebench_hard_{_safe_id(original_id)}",
                source="bigcodebench_hard",
                source_version=source_version,
                track="coding",
                verifier="bigcodebench",
                budget_grid=self.config.budget_grid or DEFAULT_BUDGET_GRID,
                prompt=prompt,
                external_id=original_id,
                metadata={
                    "original_task_id": original_id,
                    "split_name": split_name,
                    "package_version": package_version,
                    "raw_index": idx,
                    "difficulty": row.get("difficulty") or row.get("level"),
                    "hard_tagged": _is_hard(row),
                    "paper_role": "paper_extension",
                    "chat_completion_compatible": True,
                    "requires_docker": False,
                },
                external_eval={
                    "harness": "bigcodebench",
                    "task_id": original_id,
                    "split": split_name,
                    "source_version": source_version,
                },
            )
            tasks.append(task)
        if not tasks:
            raise RuntimeError("BigCodeBench-Hard loader found no tasks.")
        return tasks

    def _load_rows(self) -> tuple[list[dict[str, Any]], str, str, str | None]:
        local_path = self.config.kwargs.get("path") or self.config.kwargs.get("local_export")
        if local_path:
            path = Path(str(local_path))
            if not path.exists():
                raise RuntimeError(f"BigCodeBench-Hard local export not found: {path}")
            rows = _read_records(path)
            return rows, str(path), self.config.split or "local", None

        package_rows = _load_with_bigcodebench_package()
        if package_rows:
            rows, package_version = package_rows
            return rows, "bigcodebench-package", self.config.split or "hard", package_version

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `bigcodebench` or `datasets` to load BigCodeBench-Hard.") from exc

        dataset_name = self.config.kwargs.get("dataset") or self.config.kwargs.get("hf_dataset") or "bigcode/bigcodebench-hard"
        split = self.config.split or self.config.kwargs.get("split") or "test"
        subset = self.config.kwargs.get("subset")
        errors: list[str] = []
        candidates = [
            (dataset_name, subset, split),
            ("bigcode/bigcodebench-hard", None, split),
            ("bigcode/bigcodebench", "hard", split),
            ("bigcode/bigcodebench", None, "v0.1.4"),
        ]
        for candidate_dataset, candidate_subset, candidate_split in candidates:
            try:
                ds = (
                    load_dataset(candidate_dataset, candidate_subset, split=candidate_split)
                    if candidate_subset
                    else load_dataset(candidate_dataset, split=candidate_split)
                )
                return [dict(row) for row in ds], candidate_dataset, str(candidate_split), _version("datasets")
            except Exception as exc:  # noqa: BLE001 - try documented fallback shapes.
                errors.append(f"{candidate_dataset}/{candidate_subset or '-'}:{candidate_split}: {exc}")
        raise RuntimeError("Could not load BigCodeBench-Hard from Hugging Face. " + " | ".join(errors[:3]))


def load_bigcodebench_hard_tasks(limit: int | None = None) -> list[TaskRecord]:
    from budget2success.datasets.base import AdapterConfig

    adapter = BigCodeBenchHardAdapter(
        AdapterConfig(
            name="bigcodebench_hard",
            split="test",
            limit=limit,
            budget_grid=DEFAULT_BUDGET_GRID,
            kwargs={},
        )
    )
    return adapter.load_tasks()


def _load_with_bigcodebench_package() -> tuple[list[dict[str, Any]], str | None] | None:
    try:
        import bigcodebench  # type: ignore[import-not-found]
    except ImportError:
        return None
    package_version = _version("bigcodebench")
    try:
        from bigcodebench.data import get_bigcodebench  # type: ignore[import-not-found]

        rows = _coerce_rows(get_bigcodebench(subset="hard"))
        if rows:
            return rows, package_version
    except Exception:
        pass
    candidate_modules = [bigcodebench]
    for module_name in ("bigcodebench.data", "bigcodebench.dataset"):
        try:
            module = __import__(module_name, fromlist=["dummy"])
            candidate_modules.append(module)
        except Exception:
            continue
    for module in candidate_modules:
        for function_name in ("get_bigcodebench_hard", "load_bigcodebench_hard", "get_hard_tasks"):
            function = getattr(module, function_name, None)
            if callable(function):
                rows = _coerce_rows(function())
                if rows:
                    return rows, package_version
    return None


def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if all(isinstance(value, dict) for value in payload.values()):
            return [{**value, "task_id": key} for key, value in payload.items()]
        for key in ("tasks", "data", "rows"):
            if key in payload:
                return _coerce_rows(payload[key])
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return [dict(row) for row in payload if isinstance(row, dict)]
    return []


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _coerce_rows(payload)
    if not rows:
        raise RuntimeError(f"No BigCodeBench-Hard records found in {path}")
    return rows


def _task_id(row: dict[str, Any], idx: int) -> str:
    for key in ("task_id", "complete_prompt_id", "problem_id", "id"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return str(idx)


def _format_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("complete_prompt") or row.get("instruct_prompt") or row.get("prompt") or row.get("instruction") or ""
    signature = row.get("function_signature") or row.get("signature") or row.get("entry_point")
    imports = row.get("import_context") or row.get("libs") or row.get("required_imports")
    parts = ["You are solving a BigCodeBench-Hard Python programming task.", "", str(prompt).strip()]
    if imports:
        parts.extend(["", "Available or required imports/context:", str(imports).strip()])
    if signature:
        parts.extend(["", "Required function signature or entry point:", str(signature).strip()])
    parts.extend(["", "Return only the complete Python solution code."])
    return "\n".join(part for part in parts if part is not None).strip()


def _is_hard(row: dict[str, Any]) -> bool:
    values: list[Any] = [row.get("difficulty"), row.get("level"), row.get("split"), row.get("subset")]
    tags = row.get("tags") or row.get("labels")
    if tags is None and isinstance(row.get("metadata"), dict):
        tags = row["metadata"].get("tags")
    if isinstance(tags, list):
        values.extend(tags)
    elif tags is not None:
        values.append(tags)
    return any("hard" in str(value).lower() for value in values if value is not None)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
