from __future__ import annotations

import contextlib
import re
import signal
from types import FrameType
from typing import Iterator

from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    blocks = _CODE_BLOCK_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return text.strip()


@contextlib.contextmanager
def _alarm_timeout(seconds: float) -> Iterator[None]:
    """Small POSIX timeout helper for toy in-process verification."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum: int, _frame: FrameType | None) -> None:
        raise TimeoutError("verification timeout")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class PythonUnitTestVerifier(Verifier):
    """Local verifier for toy/dev Python tasks.

    This verifier intentionally executes code in-process so repository smoke tests
    are fast. It is **not** a sandbox and must not be used for untrusted code in
    production runs. Published coding tracks should use official harnesses such
    as EvalPlus, BigCodeBench, LiveCodeBench, or SWE-bench.
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        tests = task.metadata.get("tests")
        if not tests:
            return VerificationResult.error(error="missing_tests")
        code = extract_python_code(solution)
        program = f"{code}\n\n{tests}\n"
        namespace: dict[str, object] = {}
        try:
            with _alarm_timeout(self.timeout_seconds):
                exec(compile(program, filename=f"<budget2success:{task.task_id}>", mode="exec"), namespace, namespace)
        except TimeoutError:
            return VerificationResult.fail(error="timeout", timeout_seconds=self.timeout_seconds)
        except BaseException as exc:  # noqa: BLE001 - verifier must report candidate failures.
            return VerificationResult(
                status="failure",
                success=False,
                details={"error": type(exc).__name__, "message": str(exc)[-1000:]},
            )
        return VerificationResult(status="success", success=True, details={})


# Backwards-compatible alias used by the original skeleton.
CodingVerifier = PythonUnitTestVerifier
