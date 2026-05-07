from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from budget2success.execution.bigcodebench_bridge import BigCodeBenchBridge
from budget2success.execution.coding_verifier import PythonUnitTestVerifier
from budget2success.execution.coding_verifier import extract_python_code
from budget2success.schemas.records import TaskRecord, VerificationResult


def verify_bigcodebench(task: TaskRecord, completion: str, timeout_seconds: float = 120.0) -> VerificationResult:
    """Verify one BigCodeBench completion through the official evaluator when available.

    Unit tests in task metadata are accepted only for local smoke tests. Paper
    metrics should use labels produced by the official BigCodeBench evaluator.
    """
    if task.metadata.get("tests"):
        result = PythonUnitTestVerifier(timeout_seconds=min(timeout_seconds, 10.0)).verify(task, completion)
        result.metadata = {**result.metadata, "label_source": "local_smoke_only"}
        return result

    direct_result = _verify_with_official_package(task, completion)
    if direct_result is not None:
        return direct_result

    if importlib.util.find_spec("bigcodebench") is None and not os.getenv("BIGCODEBENCH_EVAL_COMMAND"):
        return VerificationResult.error(
            error="official_harness_unavailable",
            message="Install `bigcodebench` or set BIGCODEBENCH_EVAL_COMMAND for official BigCodeBench verification.",
            verifier_name="bigcodebench",
            task_id=task.task_id,
            metadata={
                "label_source": "official_harness_unavailable",
                "exclude_from_main_metrics": True,
                "official_harness_required": "bigcodebench",
            },
        )

    with tempfile.TemporaryDirectory(prefix="bigcodebench_") as tmp:
        tmp_path = Path(tmp)
        bridge = BigCodeBenchBridge(tmp_path)
        predictions_path = bridge.write_predictions(
            [{"task_id": task.external_id or task.task_id, "solution": completion}],
            filename="bigcodebench_predictions.jsonl",
        )
        command = _evaluation_command(predictions_path, tmp_path)
        try:
            completed = subprocess.run(
                command,
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
                verifier_name="bigcodebench",
                metadata={"label_source": "official_bigcodebench"},
            )

        success_by_task = _parse_success(bridge, tmp_path, completed.stdout)
        task_key = task.external_id or task.task_id
        if task_key in success_by_task:
            success = bool(success_by_task[task_key])
            details: dict[str, Any] = {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
                "verifier_name": "bigcodebench",
                "task_id": task_key,
            }
            if success:
                return VerificationResult.ok(**details, metadata={"label_source": "official_bigcodebench"})
            return VerificationResult.fail(**details, metadata={"label_source": "official_bigcodebench"})

        return VerificationResult.error(
            error="official_result_missing",
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            verifier_name="bigcodebench",
            task_id=task_key,
            metadata={
                "label_source": "official_harness_unavailable",
                "exclude_from_main_metrics": True,
                "official_harness_required": "bigcodebench",
            },
        )


def _evaluation_command(predictions_path: Path, output_dir: Path) -> list[str]:
    template = os.getenv("BIGCODEBENCH_EVAL_COMMAND")
    if template:
        return [
            part.replace("{predictions}", str(predictions_path)).replace("{output_dir}", str(output_dir))
            for part in template.split()
        ]
    return [
        sys.executable,
        "-m",
        "bigcodebench.evaluate",
        "--samples",
        str(predictions_path),
        "--out_dir",
        str(output_dir),
    ]


def _verify_with_official_package(task: TaskRecord, completion: str) -> VerificationResult | None:
    try:
        from bigcodebench.eval import PASS, untrusted_check  # type: ignore[import-not-found]
    except Exception:
        return None
    task_key = task.external_id or task.task_id
    problem = _official_bigcodebench_problem(task_key)
    if problem is None:
        return None
    solution = extract_python_code(completion)
    try:
        status, details = untrusted_check(
            solution,
            problem["test"],
            problem["entry_point"],
            30 * 1024,
            30 * 1024,
            10,
            1,
            20,
        )
    except Exception as exc:  # noqa: BLE001 - official evaluator failed on this sample.
        return VerificationResult.error(
            error="official_bigcodebench_exception",
            message=str(exc),
            verifier_name="bigcodebench",
            task_id=task_key,
            metadata={
                "label_source": "official_bigcodebench",
                "official_harness_required": "bigcodebench",
            },
        )
    metadata = {"label_source": "official_bigcodebench", "official_harness_required": "bigcodebench"}
    payload = {
        "status": str(status),
        "details": _json_safe(details),
        "verifier_name": "bigcodebench",
        "task_id": task_key,
    }
    if status == PASS:
        return VerificationResult.ok(**payload, metadata=metadata)
    return VerificationResult.fail(**payload, metadata=metadata)


@lru_cache(maxsize=1)
def _official_bigcodebench_hard() -> dict[str, dict[str, Any]]:
    from bigcodebench.data import get_bigcodebench  # type: ignore[import-not-found]

    return dict(get_bigcodebench(subset="hard"))


def _official_bigcodebench_problem(task_id: str) -> dict[str, Any] | None:
    try:
        return _official_bigcodebench_hard().get(task_id)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)


def _parse_success(bridge: BigCodeBenchBridge, output_dir: Path, stdout: str) -> dict[str, bool]:
    try:
        return bridge.parse_official_results(output_dir)
    except Exception:
        pass
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        try:
            return bridge.parse_official_results(_write_stdout_payload(output_dir, payload))
        except Exception:
            continue
    return {}


def _write_stdout_payload(output_dir: Path, payload: Any) -> Path:
    path = output_dir / "stdout_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
