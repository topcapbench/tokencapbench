from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class TauBenchAdapter(BenchmarkSourceAdapter):
    """Adapter placeholder for tau2-bench/tau-bench task exports."""

    source_name = "tau2"

    @classmethod
    def available(cls) -> bool:
        return True

    def load_tasks(self) -> list[TaskRecord]:
        path = self.config.kwargs.get("path")
        if not path:
            raise RuntimeError(
                "TauBenchAdapter requires a local tau2-bench export path in kwargs.path. "
                "Use a JSONL export with TokenCapBench TaskRecord fields."
            )
        from budget2success.datasets.loaders.local_jsonl import LocalJSONLAdapter

        local_cfg = self.config
        local_cfg.kwargs = {**local_cfg.kwargs, "path": path}
        tasks = LocalJSONLAdapter(local_cfg).load_tasks()
        for task in tasks:
            task.track = "agentic"
            task.source = "tau2"
            task.verifier = task.verifier or "tau2"
        return tasks
