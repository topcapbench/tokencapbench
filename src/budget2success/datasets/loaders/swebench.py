from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class SWEBenchAdapter(BenchmarkSourceAdapter):
    """Adapter for SWE-bench metadata via Hugging Face datasets.

    Verification should be delegated to the official SWE-bench harness.
    """

    source_name = "swebench"

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
            raise RuntimeError("Install `datasets` to load SWE-bench metadata.") from exc

        dataset_name = self.config.kwargs.get("dataset", "SWE-bench/SWE-bench_Lite")
        split = self.config.split or "test"
        limit = self.config.limit
        ds = load_dataset(dataset_name, split=split)
        tasks: list[TaskRecord] = []
        for idx, row in enumerate(ds):
            instance_id = row.get("instance_id") or str(idx)
            prompt = _format_swe_prompt(row)
            task = TaskRecord(
                task_id=f"swebench_{instance_id}",
                track="swe",
                source="swebench",
                source_version=dataset_name,
                external_id=instance_id,
                prompt=prompt,
                verifier="swebench",
                budget_grid=self.config.budget_grid,
                metadata={
                    "repo": row.get("repo"),
                    "base_commit": row.get("base_commit"),
                    "problem_statement": row.get("problem_statement"),
                },
                external_eval={"harness": "swebench", "dataset": dataset_name, "split": split, "instance_id": instance_id},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break
        return tasks


def _format_swe_prompt(row: dict) -> str:
    return (
        f"Repository: {row.get('repo')}\n"
        f"Base commit: {row.get('base_commit')}\n\n"
        f"Issue:\n{row.get('problem_statement', '')}\n\n"
        "Produce a patch that resolves the issue."
    )
