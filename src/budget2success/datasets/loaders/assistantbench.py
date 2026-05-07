from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class AssistantBenchAdapter(BenchmarkSourceAdapter):
    """Adapter boundary for AssistantBench exports.

    AssistantBench task formats can change with the upstream repo and web-agent
    setup. For reproducibility, export a pinned subset from the official repo to
    TokenCapBench TaskRecord JSONL, then load it through this adapter.
    """

    source_name = "assistantbench"

    @classmethod
    def available(cls) -> bool:
        return True

    def load_tasks(self) -> list[TaskRecord]:
        path = self.config.kwargs.get("path")
        if not path:
            raise RuntimeError(
                "AssistantBenchAdapter requires a local pinned AssistantBench export in kwargs.path. "
                "Use a JSONL export with TokenCapBench TaskRecord fields."
            )
        from budget2success.datasets.loaders.local_jsonl import LocalJSONLAdapter

        local_cfg = self.config
        local_cfg.kwargs = {**local_cfg.kwargs, "path": path}
        tasks = LocalJSONLAdapter(local_cfg).load_tasks()
        for task in tasks:
            task.track = "agentic"
            task.source = "assistantbench"
            task.verifier = task.verifier or "assistantbench"
        return tasks
