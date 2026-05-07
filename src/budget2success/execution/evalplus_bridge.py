from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from budget2success.execution.external_harness import ExternalHarnessResult, run_command
from budget2success.schemas.records import BudgetRunRecord, TaskRecord


class EvalPlusBridge:
    """Export EvalPlus samples and invoke the official EvalPlus evaluator."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_samples(self, samples: Iterable[dict], filename: str = "evalplus_samples.jsonl") -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def write_samples_from_records(
        self,
        tasks: Iterable[TaskRecord],
        outcomes: Iterable[BudgetRunRecord],
        filename: str = "evalplus_samples.jsonl",
    ) -> Path:
        task_by_id = {task.task_id: task for task in tasks}
        samples: list[dict] = []
        for outcome in outcomes:
            task = task_by_id.get(outcome.task_id)
            if task is None:
                continue
            samples.append({"task_id": task.external_id or task.task_id, "solution": outcome.solution})
        return self.write_samples(samples, filename=filename)

    def run_evaluation(
        self,
        samples_path: str | Path,
        dataset: str = "humaneval",
        command: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ExternalHarnessResult:
        if command is None:
            command = ["python", "-m", "evalplus.evaluate", dataset, "--samples", str(samples_path)]
        return run_command(list(command), timeout_seconds=timeout_seconds)
