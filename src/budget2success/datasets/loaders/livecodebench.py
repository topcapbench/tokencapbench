from __future__ import annotations

from pathlib import Path

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


class LiveCodeBenchAdapter(BenchmarkSourceAdapter):
    """Adapter shell for LiveCodeBench code-generation tasks."""

    source_name = "livecodebench"

    @classmethod
    def available(cls) -> bool:
        try:
            import datasets  # noqa: F401
            return True
        except ImportError:
            return False

    def load_tasks(self) -> list[TaskRecord]:
        local_path = self.config.kwargs.get("path") or self.config.kwargs.get("local_export") or self.config.kwargs.get("local_export_path")
        if local_path:
            return self._load_local_export(Path(str(local_path)))
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install `datasets` to load LiveCodeBench metadata.") from exc

        split = self.config.split or "test"
        dataset_name = self.config.kwargs.get("dataset", "livecodebench/code_generation")
        limit = self.config.limit
        ds = load_dataset(dataset_name, split=split)
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(ds):
            task_id = str(row.get("question_id") or row.get("contest_id") or idx)
            prompt = row.get("question_content") or row.get("prompt") or row.get("question") or ""
            task = TaskRecord(
                task_id=f"livecodebench_{task_id}",
                track="coding",
                source="livecodebench",
                source_version=dataset_name,
                external_id=task_id,
                prompt=prompt,
                verifier="livecodebench",
                budget_grid=self.config.budget_grid,
                metadata={
                    "platform": row.get("platform"),
                    "contest_date": row.get("contest_date"),
                    "difficulty": row.get("difficulty"),
                },
                external_eval={"harness": "livecodebench", "task_id": task_id},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks

    def _load_local_export(self, path: Path) -> list[TaskRecord]:
        if not path.exists():
            raise RuntimeError(f"Local LiveCodeBench export not found: {path}")
        limit = self.config.limit
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(read_jsonl(path)):
            task_id = str(row.get("question_id") or row.get("task_id") or row.get("contest_id") or idx)
            prompt = row.get("question_content") or row.get("prompt") or row.get("question") or ""
            external_eval = {
                "harness": "livecodebench",
                "source": "local_official_export",
                "task_id": task_id,
                "local_export_path": str(path),
            }
            for key in ["input_output", "starter_code", "contest_date", "platform", "difficulty"]:
                if key in row:
                    external_eval[key] = row.get(key)
            tasks.append(
                TaskRecord(
                    task_id=f"livecodebench_{task_id}",
                    track="coding",
                    source="livecodebench",
                    source_version=str(row.get("source_version") or path.name),
                    external_id=task_id,
                    prompt=str(prompt),
                    verifier="livecodebench",
                    budget_grid=self.config.budget_grid,
                    fresh_split=str(row.get("fresh_split") or "livecodebench_local_export"),
                    verifier_policy="official_livecodebench_required",
                    metadata={
                        "platform": row.get("platform"),
                        "contest_date": row.get("contest_date"),
                        "difficulty": row.get("difficulty"),
                        "paper_role": row.get("paper_role", "optional_fresh_coding"),
                    },
                    external_eval=external_eval,
                )
            )
            if limit and len(tasks) >= limit:
                break
        return tasks
