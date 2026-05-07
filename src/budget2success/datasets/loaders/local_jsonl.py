from __future__ import annotations

import json
from pathlib import Path

from budget2success.datasets.base import AdapterConfig, BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


class LocalJSONLAdapter(BenchmarkSourceAdapter):
    source_name = "local"

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__(config)
        self.path = self.config.kwargs.get("path")
        if self.path is None and self.config.kwargs.get("input"):
            self.path = self.config.kwargs["input"]

    def load_tasks(self) -> list[TaskRecord]:
        if not self.path:
            raise RuntimeError("LocalJSONLAdapter requires kwargs.path")
        tasks: list[TaskRecord] = []
        with Path(self.path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = TaskRecord.model_validate(json.loads(line))
                if self.config.budget_grid and task.budget_grid is None:
                    task.budget_grid = self.config.budget_grid
                tasks.append(task)
                if self.config.limit and len(tasks) >= self.config.limit:
                    break
        return tasks
