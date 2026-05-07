from __future__ import annotations

import re

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord

_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


class HendrycksMATHAdapter(BenchmarkSourceAdapter):
    """Adapter for Hendrycks MATH via Hugging Face datasets when available."""

    source_name = "math"

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
            raise RuntimeError("Install `datasets` to load MATH, or use local JSONL tasks.") from exc

        split = self.config.split or "test"
        limit = self.config.limit
        dataset_name = self.config.kwargs.get("dataset", "HuggingFaceH4/MATH-500")
        subset = self.config.kwargs.get("subset")
        ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(ds):
            solution = row.get("solution", "")
            answer = _extract_boxed(solution) or row.get("answer")
            external_id = str(row.get("unique_id") or idx)
            task = TaskRecord(
                task_id=f"math_{split}_{idx}",
                track="math",
                source="hendrycks_math",
                source_version=dataset_name,
                external_id=external_id,
                prompt=row.get("problem") or row.get("question") or "",
                answer=answer,
                verifier="math_verify_optional",
                budget_grid=self.config.budget_grid,
                metadata={"type": row.get("type") or row.get("subject"), "level": row.get("level"), "solution": solution},
                external_eval={"harness": "math_verify_optional", "dataset": dataset_name, "split": split, "external_id": external_id},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks


def _extract_boxed(solution: str) -> str | None:
    matches = _BOX_RE.findall(solution or "")
    if matches:
        return matches[-1].strip()
    return None
