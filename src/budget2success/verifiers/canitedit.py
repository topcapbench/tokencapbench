from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from budget2success.execution.coding_verifier import extract_python_code
from budget2success.schemas.records import TaskRecord, VerificationResult


def verify_canitedit(task: TaskRecord, completion: str, timeout_seconds: float = 30.0) -> VerificationResult:
    """Run a CanItEdit edited Python program against the provided test block."""
    tests = str(task.metadata.get("tests") or "")
    if not tests.strip():
        return VerificationResult.error(
            error="canitedit_tests_unavailable",
            verifier_name="canitedit",
            task_id=task.task_id,
            metadata={
                "label_source": "canitedit_tests_unavailable",
                "exclude_from_main_metrics": True,
                "official_harness_status": "hidden_tests_unavailable",
            },
        )

    edited_program = extract_python_code(completion)
    with tempfile.TemporaryDirectory(prefix="canitedit_") as tmp:
        tmp_path = Path(tmp)
        program_path = tmp_path / "candidate.py"
        program_path.write_text(f"{edited_program.rstrip()}\n\n{tests.rstrip()}\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(program_path)],
                cwd=tmp_path,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return VerificationResult.fail(
                error="timeout",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timeout_seconds=timeout_seconds,
                verifier_name="canitedit",
                metadata={"label_source": "canitedit_provided_tests"},
            )

    details: dict[str, Any] = {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "verifier_name": "canitedit",
    }
    if completed.returncode == 0:
        return VerificationResult.ok(**details, metadata={"label_source": "canitedit_provided_tests"})
    return VerificationResult.fail(**details, metadata={"label_source": "canitedit_provided_tests"})
