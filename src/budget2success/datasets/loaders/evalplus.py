from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class EvalPlusAdapter(BenchmarkSourceAdapter):
    """Adapter shell for HumanEval+/MBPP+ through EvalPlus."""

    source_name = "evalplus"

    @classmethod
    def available(cls) -> bool:
        try:
            import evalplus  # noqa: F401
            return True
        except ImportError:
            return False

    def load_tasks(self) -> list[TaskRecord]:
        try:
            from evalplus.data import get_human_eval_plus, get_mbpp_plus
        except ImportError as exc:
            raise RuntimeError("Install `evalplus` to load HumanEval+/MBPP+.") from exc

        dataset = self.config.kwargs.get("dataset", "humaneval")
        limit = self.config.limit
        raw = get_mbpp_plus() if dataset.lower() == "mbpp" else get_human_eval_plus()
        tasks: list[TaskRecord] = []
        for i, (task_id, row) in enumerate(raw.items()):
            prompt = row.get("prompt", "")
            entry_point = row.get("entry_point")
            task = TaskRecord(
                task_id=f"evalplus_{task_id}",
                track="coding",
                source=f"evalplus_{dataset}",
                source_version="evalplus",
                external_id=task_id,
                prompt=prompt,
                verifier="evalplus",
                budget_grid=self.config.budget_grid,
                metadata={"entry_point": entry_point, "raw_task_id": task_id},
                external_eval={"harness": "evalplus", "dataset": dataset, "task_id": task_id},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks
