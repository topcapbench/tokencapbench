from __future__ import annotations

from pathlib import Path

from budget2success.execution.external_harness import ExternalHarnessResult, run_command


class TauBenchBridge:
    """Command bridge for tau2-bench/tau-bench experiments."""

    def __init__(self, repo_dir: str | Path):
        self.repo_dir = Path(repo_dir)

    def run_evaluation(self, config_path: str | Path, timeout_seconds: float | None = None) -> ExternalHarnessResult:
        command = ["python", "-m", "tau2.run", "--config", str(config_path)]
        return run_command(command, cwd=self.repo_dir, timeout_seconds=timeout_seconds)
