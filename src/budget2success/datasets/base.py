from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from budget2success.schemas.records import TaskRecord


@dataclass
class AdapterConfig:
    name: str
    split: str | None = None
    limit: int | None = None
    budget_grid: list[int] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class BenchmarkSourceAdapter(ABC):
    source_name: str = "base"

    def __init__(self, config: AdapterConfig | None = None):
        self.config = config or AdapterConfig(name=self.source_name)

    @classmethod
    def available(cls) -> bool:
        return True

    @abstractmethod
    def load_tasks(self) -> list[TaskRecord]:
        raise NotImplementedError


def require_package(package: str, install_hint: str) -> None:
    try:
        __import__(package)
    except ImportError as exc:
        raise RuntimeError(f"Missing optional dependency '{package}'. {install_hint}") from exc
