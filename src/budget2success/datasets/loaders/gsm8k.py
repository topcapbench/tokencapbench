from __future__ import annotations

from budget2success.datasets.base import AdapterConfig, BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class GSM8KAdapter(BenchmarkSourceAdapter):
    """Adapter for GSM8K via Hugging Face datasets.

    Uses `openai/gsm8k` when available. The official GitHub repo is
    openai/grade-school-math; many evaluation harnesses load the same data from
    Hugging Face.
    """

    source_name = "gsm8k"

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
            raise RuntimeError("Install `datasets` to load GSM8K, or use local JSONL tasks.") from exc

        split = self.config.split or "test"
        limit = self.config.limit
        ds = load_dataset("openai/gsm8k", "main", split=split)
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(ds):
            answer = str(row["answer"]).split("####")[-1].strip()
            task = TaskRecord(
                task_id=f"gsm8k_{split}_{idx}",
                track="math",
                source="gsm8k",
                source_version="openai/gsm8k",
                external_id=str(idx),
                prompt=row["question"],
                answer=answer,
                verifier="numeric_exact",
                budget_grid=self.config.budget_grid,
                metadata={"raw_answer": row["answer"]},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks
