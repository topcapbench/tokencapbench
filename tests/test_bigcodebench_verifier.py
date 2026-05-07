from budget2success.schemas.records import TaskRecord
from budget2success.verifiers.bigcodebench import verify_bigcodebench


def test_bigcodebench_verifier_uses_local_tests_for_smoke_tasks():
    task = TaskRecord(
        task_id="bigcodebench_hard_smoke",
        track="coding",
        source="bigcodebench_hard",
        prompt="p",
        verifier="bigcodebench",
        metadata={"tests": "assert add_one(1) == 2"},
        external_eval={"harness": "bigcodebench", "task_id": "smoke"},
    )

    result = verify_bigcodebench(task, "def add_one(x):\n    return x + 1\n")

    assert result.success
    assert result.metadata["label_source"] == "local_smoke_only"


def test_bigcodebench_verifier_marks_missing_official_harness_not_main(monkeypatch):
    monkeypatch.delenv("BIGCODEBENCH_EVAL_COMMAND", raising=False)
    monkeypatch.setattr("budget2success.verifiers.bigcodebench.importlib.util.find_spec", lambda name: None)
    task = TaskRecord(
        task_id="bigcodebench_hard_1",
        track="coding",
        source="bigcodebench_hard",
        prompt="p",
        verifier="bigcodebench",
        external_eval={"harness": "bigcodebench", "task_id": "1"},
    )

    result = verify_bigcodebench(task, "def task_func():\n    pass\n")

    assert not result.success
    assert result.metadata["exclude_from_main_metrics"] is True
    assert result.metadata["official_harness_required"] == "bigcodebench"
