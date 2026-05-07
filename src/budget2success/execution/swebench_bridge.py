from __future__ import annotations

import json
from pathlib import Path

from budget2success.execution.external_harness import ExternalHarnessResult, run_command
from budget2success.schemas.records import BudgetRunRecord, TaskRecord


class SWEBenchBridge:
    """Bridge to official SWE-bench harness.

    This bridge writes prediction files in SWE-bench format and can call the
    official harness if the `swebench` package/repo is installed in the env.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_predictions(self, predictions: list[dict], filename: str = "swe_predictions.jsonl") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def write_predictions_from_records(
        self,
        tasks: list[TaskRecord],
        outcomes: list[BudgetRunRecord],
        model_name_or_path: str,
        filename: str = "swe_predictions.jsonl",
    ) -> Path:
        task_by_id = {task.task_id: task for task in tasks}
        predictions: list[dict] = []
        for outcome in outcomes:
            task = task_by_id.get(outcome.task_id)
            if task is None:
                continue
            predictions.append(
                {
                    "instance_id": task.external_id or task.task_id,
                    "model_name_or_path": model_name_or_path,
                    "model_patch": outcome.solution,
                }
            )
        return self.write_predictions(predictions, filename=filename)

    def run_evaluation(
        self,
        predictions_path: str | Path,
        dataset_name: str = "SWE-bench/SWE-bench_Lite",
        split: str = "test",
        run_id: str = "budget2success",
        timeout_seconds: float | None = None,
    ) -> ExternalHarnessResult:
        command = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--predictions_path",
            str(predictions_path),
            "--run_id",
            run_id,
        ]
        return run_command(command, timeout_seconds=timeout_seconds)
