from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from budget2success.execution.external_harness import ExternalHarnessResult, run_command
from budget2success.schemas.records import BudgetRunRecord, TaskRecord, VerificationResult


class BigCodeBenchBridge:
    """Export BigCodeBench generations and invoke an installed official evaluator."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_predictions(self, predictions: Iterable[dict], filename: str = "bigcodebench_predictions.jsonl") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def write_predictions_from_records(
        self,
        tasks: Iterable[TaskRecord],
        outcomes: Iterable[BudgetRunRecord | dict[str, Any]],
        filename: str = "bigcodebench_predictions.jsonl",
    ) -> Path:
        task_by_id = {task.task_id: task for task in tasks}
        predictions: list[dict] = []
        for raw_outcome in outcomes:
            outcome = _outcome_dict(raw_outcome)
            task = task_by_id.get(str(outcome.get("task_id")))
            if task is None:
                continue
            self._require_official_metadata(task)
            predictions.append({"task_id": task.external_id or task.task_id, "solution": str(outcome.get("solution") or "")})
        return self.write_predictions(predictions, filename=filename)

    def write_predictions_grouped_by_budget(
        self,
        tasks: Iterable[TaskRecord],
        outcomes: Iterable[BudgetRunRecord | dict[str, Any]],
        output_root: str | Path,
    ) -> dict[int, Path]:
        task_by_id = {task.task_id: task for task in tasks}
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for raw_outcome in outcomes:
            outcome = _outcome_dict(raw_outcome)
            task = task_by_id.get(str(outcome.get("task_id")))
            if task is None:
                continue
            self._require_official_metadata(task)
            grouped[int(outcome["budget"])].append(
                {"task_id": task.external_id or task.task_id, "solution": str(outcome.get("solution") or "")}
            )
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[int, Path] = {}
        for budget, predictions in sorted(grouped.items()):
            path = root / f"bigcodebench_predictions_budget_{budget}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for prediction in predictions:
                    f.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
            paths[budget] = path
        return paths

    def run_evaluation(
        self,
        predictions_path: str | Path,
        command: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ExternalHarnessResult:
        if command is None:
            command = ["python", "-m", "bigcodebench.evaluate", "--samples", str(predictions_path)]
        return run_command(list(command), timeout_seconds=timeout_seconds)

    def parse_official_results(self, result_path: str | Path) -> dict[str, bool]:
        path = Path(result_path)
        if path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".json", ".jsonl"}
                and "prediction" not in candidate.name.lower()
            )
            errors: list[str] = []
            for candidate in candidates:
                try:
                    return self.parse_official_results(candidate)
                except ValueError as exc:
                    errors.append(f"{candidate}: {exc}")
            raise ValueError("No parseable BigCodeBench official result files found. " + " | ".join(errors[:5]))
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            payload: Any = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        parsed = _parse_result_payload(payload)
        if not parsed:
            raise ValueError(f"No BigCodeBench pass/fail labels found in {path}")
        return parsed

    def merge_official_results_into_outcomes(
        self,
        outcomes: Iterable[BudgetRunRecord | dict[str, Any]],
        tasks: Iterable[TaskRecord],
        success_by_task: dict[str, bool],
    ) -> list[dict[str, Any]]:
        task_by_id = {task.task_id: task for task in tasks}
        merged: list[dict[str, Any]] = []
        for raw_outcome in outcomes:
            outcome = _outcome_dict(raw_outcome)
            task = task_by_id.get(str(outcome.get("task_id")))
            if task is None:
                continue
            self._require_official_metadata(task)
            task_id = task.external_id or task.task_id
            if task_id not in success_by_task:
                raise ValueError(f"Missing official BigCodeBench label for task_id={task_id}")
            success = bool(success_by_task[task_id])
            outcome["success"] = success
            verification = VerificationResult.ok if success else VerificationResult.fail
            outcome["verification"] = verification(
                harness="bigcodebench",
                task_id=task_id,
                metadata={"label_source": "official_bigcodebench"},
            ).model_dump(mode="json")
            metadata = dict(outcome.get("metadata") or {})
            metadata.update(
                {
                    "label_source": "official_bigcodebench",
                    "official_harness_required": "bigcodebench",
                    "exclude_from_main_metrics": False,
                }
            )
            outcome["metadata"] = metadata
            merged.append(outcome)
        return merged

    def _require_official_metadata(self, task: TaskRecord) -> None:
        harness = str((task.external_eval or {}).get("harness") or "")
        if harness != "bigcodebench":
            raise ValueError("BigCodeBenchBridge requires official BigCodeBench harness metadata.")
        if not (task.external_id or (task.external_eval or {}).get("task_id")):
            raise ValueError("BigCodeBenchBridge requires a BigCodeBench task id.")


def _outcome_dict(outcome: BudgetRunRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(outcome, BudgetRunRecord):
        return outcome.model_dump(mode="json")
    return dict(outcome)


def _parse_result_payload(payload: Any) -> dict[str, bool]:
    if isinstance(payload, list):
        parsed: dict[str, bool] = {}
        for record in payload:
            if not isinstance(record, dict):
                continue
            pair = _parse_result_record(record)
            if pair is not None:
                parsed[pair[0]] = pair[1]
        return parsed
    if isinstance(payload, dict):
        if payload and all(isinstance(value, bool) for value in payload.values()):
            return {str(key): bool(value) for key, value in payload.items()}
        pair = _parse_result_record(payload)
        if pair is not None:
            return {pair[0]: pair[1]}
        for key in ("results", "data", "records", "rows", "eval_results", "task_results"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                parsed = _parse_result_payload(value)
                if parsed:
                    return parsed
        if payload and all(isinstance(value, dict) for value in payload.values()):
            parsed = {}
            for key, value in payload.items():
                success = _extract_success(value)
                if success is not None:
                    parsed[str(key)] = success
            return parsed
    raise ValueError("Unsupported BigCodeBench result format")


def _parse_result_record(record: dict[str, Any]) -> tuple[str, bool] | None:
    task_id = _extract_task_id(record)
    success = _extract_success(record)
    if task_id is None or success is None:
        return None
    return task_id, success


def _extract_task_id(record: dict[str, Any]) -> str | None:
    for key in ("task_id", "complete_prompt_id", "problem_id", "id"):
        value = record.get(key)
        if value not in {None, ""} and not isinstance(value, (dict, list)):
            return str(value)
    for key in ("task", "metadata"):
        value = record.get(key)
        if isinstance(value, dict):
            nested = _extract_task_id(value)
            if nested is not None:
                return nested
    return None


def _extract_success(record: dict[str, Any]) -> bool | None:
    for key in ("passed", "success", "pass", "is_correct", "correct", "all_passed"):
        if key in record:
            converted = _to_bool(record[key])
            if converted is not None:
                return converted
    for key in ("status", "result", "verdict"):
        value = record.get(key)
        if isinstance(value, str):
            converted = _status_to_bool(value)
            if converted is not None:
                return converted
    for key in ("evaluation", "eval", "details", "metadata"):
        value = record.get(key)
        if isinstance(value, dict):
            nested = _extract_success(value)
            if nested is not None:
                return nested
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        return _status_to_bool(value)
    return None


def _status_to_bool(value: str) -> bool | None:
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in {"1", "true", "yes", "pass", "passed", "success", "accepted", "correct", "ok"}:
        return True
    if normalized in {"0", "false", "no", "fail", "failed", "failure", "wrong", "wrong_answer", "error", "timeout"}:
        return False
    return None
