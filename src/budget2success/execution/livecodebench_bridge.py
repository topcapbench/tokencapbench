from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from budget2success.execution.external_harness import ExternalHarnessResult, run_command
from budget2success.schemas.records import BudgetRunRecord, TaskRecord, VerificationResult


class LiveCodeBenchBridge:
    """Export LiveCodeBench generations and call a pinned official runner command."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_predictions(self, predictions: Iterable[dict], filename: str = "livecodebench_predictions.json") -> Path:
        path = self.output_dir / filename
        rows = list(predictions)
        if path.suffix.lower() == ".jsonl":
            with path.open("w", encoding="utf-8") as f:
                for prediction in rows:
                    f.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_predictions_from_records(
        self,
        tasks: Iterable[TaskRecord],
        outcomes: Iterable[BudgetRunRecord],
        filename: str = "livecodebench_predictions.json",
    ) -> Path:
        task_by_id = {task.task_id: task for task in tasks}
        predictions: list[dict] = []
        for outcome in outcomes:
            task = task_by_id.get(outcome.task_id)
            if task is None:
                continue
            self._require_official_metadata(task)
            predictions.append({"question_id": task.external_id or task.task_id, "code_list": [_extract_python_code(outcome.solution)]})
        return self.write_predictions(predictions, filename=filename)

    def write_predictions_grouped_by_budget(
        self,
        tasks: Iterable[TaskRecord],
        outcomes: Iterable[BudgetRunRecord | dict[str, Any]],
        output_root: str | Path,
    ) -> dict[int, Path]:
        """Write one official LiveCodeBench custom-output JSON file per token budget."""

        task_by_id = {task.task_id: task for task in tasks}
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for raw_outcome in outcomes:
            outcome = _outcome_dict(raw_outcome)
            task = task_by_id.get(str(outcome.get("task_id")))
            if task is None:
                continue
            self._require_official_metadata(task)
            budget = int(outcome["budget"])
            grouped[budget].append(
                {
                    "question_id": task.external_id or str((task.external_eval or {}).get("task_id") or task.task_id),
                    "code_list": [_extract_python_code(str(outcome.get("solution") or ""))],
                }
            )

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[int, Path] = {}
        for budget, predictions in sorted(grouped.items()):
            path = root / f"livecodebench_predictions_budget_{budget}.json"
            path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            paths[budget] = path
        return paths

    def run_evaluation(
        self,
        predictions_path: str | Path,
        command: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
        cwd: str | Path | None = None,
    ) -> ExternalHarnessResult:
        if command is None:
            command = [
                "python",
                "-m",
                "lcb_runner.runner.custom_evaluator",
                "--scenario",
                "codegeneration",
                "--custom_output_file",
                str(predictions_path),
            ]
        return run_command(list(command), cwd=cwd, timeout_seconds=timeout_seconds)

    def parse_official_results(self, result_path: str | Path) -> dict[str, bool]:
        """Parse official LiveCodeBench output into question_id -> success."""

        path = Path(result_path)
        if path.is_dir():
            candidates = _result_candidates(path)
            if not candidates:
                raise ValueError(f"No JSON/JSONL LiveCodeBench result files found under {path}")
            errors: list[str] = []
            for candidate in candidates:
                try:
                    return self.parse_official_results(candidate)
                except ValueError as exc:
                    errors.append(f"{candidate}: {exc}")
            raise ValueError("Unsupported LiveCodeBench result directory. " + " | ".join(errors[:5]))
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            parsed = _parse_result_payload(records)
        else:
            parsed = _parse_result_payload(json.loads(path.read_text(encoding="utf-8")))
        if not parsed:
            raise ValueError(f"No LiveCodeBench question results found in {path}")
        return parsed

    def merge_official_results_into_outcomes(
        self,
        outcomes: Iterable[BudgetRunRecord | dict[str, Any]],
        tasks: Iterable[TaskRecord],
        success_by_budget_and_question: dict[int, dict[str, bool]],
    ) -> list[dict[str, Any]]:
        """Return outcomes with official LiveCodeBench labels merged in."""

        task_by_id = {task.task_id: task for task in tasks}
        merged: list[dict[str, Any]] = []
        for raw_outcome in outcomes:
            outcome = _outcome_dict(raw_outcome)
            task = task_by_id.get(str(outcome.get("task_id")))
            if task is None:
                continue
            self._require_official_metadata(task)
            budget = int(outcome["budget"])
            question_id = task.external_id or str((task.external_eval or {}).get("task_id") or task.task_id)
            budget_results = success_by_budget_and_question.get(budget)
            if budget_results is None or question_id not in budget_results:
                raise ValueError(
                    f"Missing official LiveCodeBench label for budget={budget}, "
                    f"question_id={question_id}, task_id={task.task_id}"
                )
            success = bool(budget_results[question_id])
            outcome["success"] = success
            outcome["verification"] = VerificationResult.ok(
                harness="livecodebench",
                question_id=question_id,
                metadata={"label_source": "official_livecodebench"},
            ).model_dump(mode="json")
            if not success:
                outcome["verification"] = VerificationResult.fail(
                    harness="livecodebench",
                    question_id=question_id,
                    metadata={"label_source": "official_livecodebench"},
                ).model_dump(mode="json")
            metadata = dict(outcome.get("metadata") or {})
            metadata.update(
                {
                    "label_source": "official_livecodebench",
                    "official_harness_required": "livecodebench",
                    "exclude_from_main_metrics": False,
                }
            )
            outcome["metadata"] = metadata
            merged.append(outcome)
        return merged

    def _require_official_metadata(self, task: TaskRecord) -> None:
        harness = str((task.external_eval or {}).get("harness") or "")
        if harness not in {"livecodebench", "lcb_runner"}:
            raise ValueError("LiveCodeBenchBridge requires official LiveCodeBench harness metadata.")
        if not (task.external_id or (task.external_eval or {}).get("task_id")):
            raise ValueError("LiveCodeBenchBridge requires a LiveCodeBench question/task id.")


def _outcome_dict(outcome: BudgetRunRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(outcome, BudgetRunRecord):
        return outcome.model_dump(mode="json")
    return dict(outcome)


def _extract_python_code(model_output: str) -> str:
    lines = model_output.splitlines()
    fence_indices = [idx for idx, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fence_indices) >= 2:
        return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]]).strip()
    return model_output.strip()


def _result_candidates(path: Path) -> list[Path]:
    excluded_names = {"livecodebench_predictions.jsonl"}
    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".json", ".jsonl"}
        and candidate.name not in excluded_names
    ]
    return sorted(candidates, key=lambda item: (not item.name.endswith("_eval_all.json"), "result" not in item.name.lower(), str(item)))


def _parse_result_payload(payload: Any) -> dict[str, bool]:
    if isinstance(payload, list):
        return _parse_result_records(payload)
    if isinstance(payload, dict):
        if payload and all(isinstance(value, bool) for value in payload.values()):
            return {str(key): bool(value) for key, value in payload.items()}
        record = _parse_single_result_record(payload)
        if record is not None:
            key, value = record
            return {key: value}
        for key in ("results", "result", "data", "records", "rows", "predictions", "outputs"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                parsed = _parse_result_payload(value)
                if parsed:
                    return parsed
        if payload and all(isinstance(value, dict) for value in payload.values()):
            parsed: dict[str, bool] = {}
            for key, value in payload.items():
                success = _extract_success(value)
                if success is not None:
                    parsed[str(key)] = success
            if parsed:
                return parsed
    raise ValueError("Unsupported LiveCodeBench result format")


def _parse_result_records(records: list[Any]) -> dict[str, bool]:
    parsed: dict[str, bool] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        pair = _parse_single_result_record(record)
        if pair is None:
            continue
        question_id, success = pair
        parsed[question_id] = success
    if not parsed:
        raise ValueError("No records with question id and pass/fail label")
    return parsed


def _parse_single_result_record(record: dict[str, Any]) -> tuple[str, bool] | None:
    question_id = _extract_question_id(record)
    success = _extract_success(record)
    if question_id is None or success is None:
        return None
    return question_id, success


def _extract_question_id(record: dict[str, Any]) -> str | None:
    for key in ("question_id", "questionId", "task_id", "taskId", "id", "problem_id", "problemId"):
        value = record.get(key)
        if value not in {None, ""} and not isinstance(value, (dict, list)):
            return str(value)
    for key in ("question", "problem", "metadata"):
        value = record.get(key)
        if isinstance(value, dict):
            nested = _extract_question_id(value)
            if nested is not None:
                return nested
    return None


def _extract_success(record: dict[str, Any]) -> bool | None:
    graded = record.get("graded_list")
    if isinstance(graded, list) and graded:
        return any(bool(item) for item in graded)
    pass_at_1 = record.get("pass@1")
    if isinstance(pass_at_1, (int, float)):
        return pass_at_1 > 0
    for key in ("success", "passed", "pass", "is_correct", "correct", "accepted", "all_passed"):
        if key in record:
            converted = _to_bool(record[key])
            if converted is not None:
                return converted
    for key in ("status", "verdict", "result"):
        value = record.get(key)
        if isinstance(value, str):
            converted = _status_to_bool(value)
            if converted is not None:
                return converted
    for key in ("evaluation", "eval", "metadata", "details", "result", "results"):
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
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "pass", "passed", "success", "accepted", "correct", "ok"}:
            return True
        if normalized in {"0", "false", "no", "n", "fail", "failed", "failure", "wrong", "wrong_answer", "error"}:
            return False
    return None


def _status_to_bool(value: str) -> bool | None:
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in {"accepted", "pass", "passed", "success", "correct", "ok"}:
        return True
    if normalized in {"wrong_answer", "failed", "failure", "runtime_error", "compile_error", "timeout", "error"}:
        return False
    return None
