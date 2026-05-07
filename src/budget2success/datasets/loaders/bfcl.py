from __future__ import annotations

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class BFCLAdapter(BenchmarkSourceAdapter):
    """Adapter placeholder for Berkeley Function Calling Leaderboard.

    BFCL files vary by release. Configure `kwargs.path` to a local JSONL/JSON
    export when the BFCL repository is checked out.
    """

    source_name = "bfcl"

    @classmethod
    def available(cls) -> bool:
        return True

    def load_tasks(self) -> list[TaskRecord]:
        path = self.config.kwargs.get("path")
        if not path:
            raise RuntimeError(
                "BFCLAdapter requires a local BFCL export path in kwargs.path. "
                "Use scripts/prepare_tasks.py after cloning the BFCL repo."
            )
        # Delegate to LocalJSONL-compatible format for now. This keeps the core
        # repo runnable while preserving a clear BFCL integration point.
        from budget2success.datasets.loaders.local_jsonl import LocalJSONLAdapter

        local_cfg = self.config
        local_cfg.kwargs = {**local_cfg.kwargs, "path": path}
        tasks = LocalJSONLAdapter(local_cfg).load_tasks()
        for task in tasks:
            task.track = "agentic"
            task.source = "bfcl"
            task.verifier = task.verifier or "bfcl"
        return tasks
