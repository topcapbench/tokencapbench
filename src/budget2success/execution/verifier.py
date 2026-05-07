from __future__ import annotations

from abc import ABC, abstractmethod

from budget2success.schemas.records import TaskRecord, VerificationResult


class Verifier(ABC):
    @abstractmethod
    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        raise NotImplementedError
