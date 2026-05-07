from budget2success.execution.math_verifier import (
    MathVerifierMode,
    MathVerifyOptionalVerifier,
    NumericExactVerifier,
    TaskAwareMathVerifier,
    classify_math_answer,
    extract_final_numeric_answer,
)
from budget2success.schemas.records import TaskRecord


def _math_task(answer: str = "42") -> TaskRecord:
    return TaskRecord(
        task_id="math_test",
        track="math",
        prompt="Find the value.",
        verifier="math_verify_optional",
        answer=answer,
        source="hendrycks_math",
        budget_grid=[128],
    )


def test_math_verify_optional_has_fallback_when_dependency_available():
    verifier = MathVerifyOptionalVerifier()

    assert hasattr(verifier, "_fallback")


def test_math_verify_optional_falls_back_on_parse_error():
    verifier = MathVerifyOptionalVerifier()

    def raise_parse(_: str):
        raise RuntimeError("parse failed")

    verifier._parse = raise_parse
    verifier._verify = lambda _gold, _pred: True

    result = verifier.verify(_math_task(), "The answer is \\boxed{42}.")

    assert result.success is True
    assert result.status == "success"
    assert result.details["math_verify_available"] is True
    assert "math_verify_error" in result.details


def test_numeric_exact_verifier_empty_candidate_is_failure_not_error():
    result = NumericExactVerifier().verify(_math_task(), "I cannot solve this.")

    assert result.success is False
    assert result.status == "failure"
    assert result.metadata["verifier_mode"] == MathVerifierMode.STRICT.value


def test_numeric_exact_strict_uses_final_answer_not_any_mentioned_number():
    result = NumericExactVerifier().verify(_math_task("4"), "We compute 2+2=4. Final answer: 5.")

    assert result.success is False
    assert result.details["mode"] == "strict"
    assert result.details["extracted"] == "5"


def test_extract_final_numeric_answer_prefers_boxed_answer():
    assert extract_final_numeric_answer("First 5, then \\boxed{42}") == "42"


def test_strict_extraction_cases():
    verifier = NumericExactVerifier(mode="strict")

    assert verifier.verify(_math_task("42"), "We get \\boxed{42}.").success
    assert verifier.verify(_math_task("42"), "Final answer: 42").success
    assert verifier.verify(_math_task("1/2"), "Final answer: 0.5").success
    assert verifier.verify(_math_task("0.5"), "Final answer: 1/2").success
    assert verifier.verify(_math_task("-7"), "After simplification, answer is -7.").success
    assert not verifier.verify(_math_task("42"), "The numbers 42 and 17 appear. Final answer: 17").success
    assert not verifier.verify(_math_task("42"), "").success


def test_classify_math_answer_tags_tuple_coordinates():
    assert classify_math_answer(r"\left(3, \frac{\pi}{2}\right)") == "tuple_or_coordinate"
    assert classify_math_answer("42") == "numeric"
    assert classify_math_answer(r"\frac{1}{2}") == "fraction"
    assert classify_math_answer(None) == "missing"


def test_task_aware_math_verifier_routes_gsm8k_to_strict_numeric():
    task = TaskRecord(
        task_id="g1",
        track="math",
        prompt="Compute.",
        verifier="numeric_exact",
        answer="4",
        source="gsm8k",
        budget_grid=[64],
    )

    result = TaskAwareMathVerifier().verify(task, "We saw 4 but final answer: 5.")

    assert result.success is False
    assert result.metadata["task_aware_policy"] == "numeric_strict_for_gsm8k_or_non_symbolic"
    assert result.metadata["source"] == "gsm8k"


def test_task_aware_math_verifier_uses_math_verify_for_symbolic_when_available():
    task = _math_task(r"x^2")
    verifier = TaskAwareMathVerifier()
    calls = {"parse": 0}

    def parse(value: str):
        calls["parse"] += 1
        return value

    verifier._math_verify._parse = parse
    verifier._math_verify._verify = lambda gold, pred: gold == pred

    result = verifier.verify(task, r"\boxed{x^2}")

    assert calls["parse"] == 2
    assert result.metadata["task_aware_policy"] == "math_verify_for_symbolic"
    assert result.metadata["math_verify_available"] is True


def test_task_aware_strict_returns_error_when_math_verify_missing_for_symbolic():
    task = _math_task(r"x^2")
    verifier = TaskAwareMathVerifier(require_math_verify_for_symbolic=True)
    verifier._math_verify._parse = None
    verifier._math_verify._verify = None

    result = verifier.verify(task, r"\boxed{x^2}")

    assert result.success is False
    assert result.status == "error"
    assert result.details["error"] == "math_verify_required"
    assert result.metadata["math_verify_available"] is False
