from __future__ import annotations

from budget2success.execution.aider_polyglot_bridge import AiderPolyglotVerifier
from budget2success.execution.coding_verifier import PythonUnitTestVerifier
from budget2success.execution.evalplus_verifier import EvalPlusOfficialVerifier
from budget2success.execution.external_verifier import ExternalCommandVerifier
from budget2success.execution.math_verifier import (
    ExactMatchMathVerifier,
    MathVerifyOptionalVerifier,
    NumericExactVerifier,
    TaskAwareMathVerifier,
)
from budget2success.execution.swe_verifier import SWEVerifier
from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult
from budget2success.verifiers.bigcodebench import verify_bigcodebench
from budget2success.verifiers.canitedit import verify_canitedit


class OfficialHarnessRequiredVerifier(Verifier):
    """Refuse local paper-grade verification when an official harness is required."""

    def __init__(self, harness_name: str):
        self.harness_name = harness_name

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        metadata = {
            "label_source": "official_harness_placeholder",
            "exclude_from_main_metrics": True,
            "official_harness_required": self.harness_name,
        }
        return VerificationResult.error(
            error="official_harness_required",
            harness=self.harness_name,
            task_id=task.task_id,
            metadata=metadata,
            message=(
                f"{self.harness_name} tasks require the official evaluator bridge; "
                "local smoke-test verification is intentionally unavailable."
            ),
        )


class FunctionVerifier(Verifier):
    def __init__(self, function):
        self.function = function

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        return self.function(task, solution)


class RecordOnlyVerifier(Verifier):
    """Record model outputs for later batch verification."""

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        return VerificationResult.error(
            error="record_only_batch_verification_pending",
            task_id=task.task_id,
            metadata={"label_source": "record_only_pending_batch_verification"},
        )


def get_verifier(name: str) -> Verifier:
    key = (name or "").lower()
    if key in {"record_only", "batch_pending", "defer_verification"}:
        return RecordOnlyVerifier()
    if key in {"exact", "exact_match", "math_exact"}:
        return ExactMatchMathVerifier()
    if key in {"numeric_exact", "numeric_exact_strict", "math_strict", "gsm8k"}:
        return NumericExactVerifier(mode="strict")
    if key in {"numeric_exact_lenient", "math_lenient"}:
        return NumericExactVerifier(mode="lenient")
    if key in {"math_verify", "math_verify_optional"}:
        return MathVerifyOptionalVerifier()
    if key in {"task_aware_math", "math_task_aware"}:
        return TaskAwareMathVerifier(require_math_verify_for_symbolic=False)
    if key in {"task_aware_strict_math"}:
        return TaskAwareMathVerifier(require_math_verify_for_symbolic=True)
    if key in {"python_unit", "python_unit_test", "coding", "local_python"}:
        return PythonUnitTestVerifier()
    if key in {"aider_polyglot", "aider_polyglot_tests"}:
        return AiderPolyglotVerifier()
    if key in {"swe", "swebench"}:
        return SWEVerifier()
    if key in {"external_command", "terminal_bench_official", "browsergym_official"}:
        return ExternalCommandVerifier()
    if key in {"evalplus", "humaneval_plus", "mbpp_plus"}:
        dataset = "mbpp" if "mbpp" in key else "humaneval"
        return EvalPlusOfficialVerifier(dataset=dataset)
    if key in {"bigcodebench", "bigcodebench_official"}:
        return FunctionVerifier(verify_bigcodebench)
    if key in {"canitedit", "can_it_edit"}:
        return FunctionVerifier(verify_canitedit)
    if key in {"livecodebench", "bfcl", "tau2", "tau_bench", "assistantbench"}:
        harness = key
        return OfficialHarnessRequiredVerifier(harness)
    raise ValueError(f"Unknown verifier '{name}'. Refusing to fall back to a weak local verifier.")
