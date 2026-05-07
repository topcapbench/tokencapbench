from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class BigCodeBenchAdapter(BenchmarkSourceAdapter):
    """Adapter shell for BigCodeBench.

    BigCodeBench has its own evaluator and Docker setup. This adapter creates
    local TaskRecords and records enough metadata for an external evaluator
    bridge to run paper-grade verification.
    """

    source_name = "bigcodebench"

    @classmethod
    def available(cls) -> bool:
        try:
            import datasets  # noqa: F401
            return True
        except ImportError:
            return False

    def load_tasks(self) -> list[TaskRecord]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` to load BigCodeBench metadata.") from exc

        split = self.config.split or "v0.1.4"
        dataset_name = self.config.kwargs.get("dataset", "bigcode/bigcodebench")
        subset = self.config.kwargs.get("subset")
        limit = self.config.limit
        ds = load_dataset(dataset_name, split=split) if subset is None else load_dataset(dataset_name, subset, split=split)
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(ds):
            task_id = str(row.get("task_id") or row.get("complete_prompt_id") or idx)
            prompt = row.get("complete_prompt") or row.get("instruct_prompt") or row.get("prompt") or ""
            task = TaskRecord(
                task_id=f"bigcodebench_{task_id}",
                track="coding",
                source="bigcodebench",
                source_version=dataset_name,
                external_id=task_id,
                prompt=prompt,
                verifier="bigcodebench",
                budget_grid=self.config.budget_grid,
                metadata={"raw_index": idx, "libs": row.get("libs"), "difficulty": row.get("difficulty")},
                external_eval={"harness": "bigcodebench", "task_id": task_id},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks
