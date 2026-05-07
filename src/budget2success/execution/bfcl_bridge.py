from __future__ import annotations

from pathlib import Path

from budget2success.execution.external_harness import ExternalHarnessResult, run_command


class BFCLBridge:
    """Command bridge for BFCL after the Gorilla/BFCL repo is installed."""

    def __init__(self, repo_dir: str | Path):
        self.repo_dir = Path(repo_dir)

    def run_evaluation(self, predictions_path: str | Path, timeout_seconds: float | None = None) -> ExternalHarnessResult:
        # BFCL command names change by version; keep this command explicit and
        # editable rather than pretending one command fits all releases.
        command = ["python", "-m", "bfcl_eval", "--predictions", str(predictions_path)]
        return run_command(command, cwd=self.repo_dir, timeout_seconds=timeout_seconds)
