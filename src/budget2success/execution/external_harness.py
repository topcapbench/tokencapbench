from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExternalHarnessResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False


def run_command(command: list[str], cwd: str | Path | None = None, timeout_seconds: float | None = None) -> ExternalHarnessResult:
    if not command:
        return ExternalHarnessResult(success=False, returncode=-2, stdout="", stderr="External harness command is empty")
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return ExternalHarnessResult(
            success=False,
            returncode=-127,
            stdout="",
            stderr=f"External harness executable not found: {command[0]} ({exc})",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExternalHarnessResult(
            success=False,
            returncode=-9,
            stdout=stdout,
            stderr=f"External harness timed out after {timeout_seconds} seconds. {stderr}".strip(),
            timed_out=True,
        )
    return ExternalHarnessResult(success=proc.returncode == 0, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
