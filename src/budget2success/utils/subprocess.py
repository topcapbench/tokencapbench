from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float | None = None


def run_command(
    command: Sequence[str],
    cwd: str | Path | None = None,
    timeout_seconds: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        timeout=timeout_seconds,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)
