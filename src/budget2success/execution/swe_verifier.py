from __future__ import annotations

from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult


class SWEVerifier(Verifier):
    """Placeholder verifier for SWE tasks.

    Paper-grade SWE results should use `SWEBenchBridge`, which exports patches
    and calls the official SWE-bench harness. This class intentionally refuses
    to mark arbitrary text as successful.
    """

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        return VerificationResult.error(
            error="swe_requires_official_harness",
            message="Use budget2success.execution.swebench_bridge.SWEBenchBridge for SWE-bench evaluation.",
        )
