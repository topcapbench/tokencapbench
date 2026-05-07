from budget2success.execution.coding_verifier import PythonUnitTestVerifier
from budget2success.execution.math_verifier import NumericExactVerifier, extract_boxed_answer
from budget2success.execution.verifier_registry import get_verifier
from budget2success.schemas.records import TaskRecord


def test_extract_boxed_answer():
    assert extract_boxed_answer("Answer is \\boxed{42}") == "42"


def test_numeric_verifier():
    task = TaskRecord(task_id="t", track="math", prompt="Compute.", verifier="numeric_exact", answer="42")
    result = NumericExactVerifier().verify(task, "The answer is 42")
    assert result.success


def test_python_unit_test_verifier():
    task = TaskRecord(
        task_id="c",
        track="coding",
        prompt="Write add_one.",
        verifier="python_unit_test",
        metadata={"tests": "assert add_one(1) == 2"},
    )
    result = PythonUnitTestVerifier().verify(task, "def add_one(x):\n    return x + 1\n")
    assert result.success


def test_official_verifier_names_do_not_fall_back_to_local_exact_match():
    verifier = get_verifier("evalplus")
    task = TaskRecord(task_id="c", track="coding", prompt="Write code.", verifier="evalplus")
    result = verifier.verify(task, "def solution(): pass")
    assert not result.success
    assert result.details["error"] in {"evalplus_task_not_found", "evalplus_unavailable"}


def test_unknown_verifier_raises_instead_of_falling_back():
    try:
        get_verifier("not_a_real_verifier")
    except ValueError as exc:
        assert "Refusing to fall back" in str(exc)
        return
    raise AssertionError("Expected ValueError")
