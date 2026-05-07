from __future__ import annotations

from pathlib import Path

from budget2success.data.task_schema import TaskRecord
from budget2success.execution.verifier import VerificationResult, Verifier
from budget2success.execution.external_harness import run_command


class ExternalCommandVerifier(Verifier):
    """Verifier that delegates success to an external command.

    Intended for wrappers around official benchmark CLIs. Expected metadata:
      verify_command: list[str]
      cwd: optional path
    """

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        command = task.metadata.get("verify_command")
        if not command:
            return VerificationResult.error(error="missing_verify_command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            return VerificationResult.error(error="invalid_verify_command", command=command)
        cwd = task.metadata.get("cwd")
        result = run_command(command, cwd=Path(cwd) if cwd else None)
        details = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
        if result.success:
            return VerificationResult.ok(**details)
        return VerificationResult.fail(**details)
