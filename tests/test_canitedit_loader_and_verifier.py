import json

from budget2success.datasets.base import AdapterConfig
from budget2success.datasets.canitedit import CanItEditAdapter
from budget2success.schemas.records import TaskRecord
from budget2success.verifiers.canitedit import verify_canitedit


def test_canitedit_loader_uses_local_descriptive_rows(tmp_path):
    export = tmp_path / "canitedit.jsonl"
    export.write_text(
        json.dumps(
            {
                "id": 10,
                "full_name": "10_add_one",
                "before": "def add_one(x):\n    return x\n",
                "after": "def add_one(x):\n    return x + 1\n",
                "tests": "assert add_one(1) == 2",
                "instruction_descriptive": "Make add_one add one.",
                "instruction_lazy": "Fix it.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = CanItEditAdapter(
        AdapterConfig(
            name="canitedit",
            limit=105,
            budget_grid=[128],
            kwargs={"path": str(export), "instruction_style": "descriptive"},
        )
    )

    tasks = adapter.load_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "canitedit_descriptive_10_add_one"
    assert tasks[0].track == "code_editing"
    assert tasks[0].source == "canitedit"
    assert tasks[0].verifier == "canitedit"
    assert tasks[0].budget_grid == [128]
    assert tasks[0].metadata["instruction"] == "Make add_one add one."
    assert tasks[0].metadata["chat_completion_compatible"] is True
    assert tasks[0].metadata["requires_docker"] is False


def test_canitedit_verifier_runs_provided_tests():
    task = TaskRecord(
        task_id="canitedit_descriptive_add_one",
        track="code_editing",
        source="canitedit",
        prompt="p",
        verifier="canitedit",
        metadata={"tests": "assert add_one(1) == 2\nassert add_one(41) == 42"},
    )

    result = verify_canitedit(task, "def add_one(x):\n    return x + 1\n")

    assert result.success
    assert result.details["verifier_name"] == "canitedit"
    assert result.metadata["label_source"] == "canitedit_provided_tests"


def test_canitedit_verifier_marks_missing_tests_not_main():
    task = TaskRecord(
        task_id="canitedit_descriptive_missing",
        track="code_editing",
        source="canitedit",
        prompt="p",
        verifier="canitedit",
    )

    result = verify_canitedit(task, "print('x')")

    assert not result.success
    assert result.metadata["exclude_from_main_metrics"] is True
    assert result.details["verifier_name"] == "canitedit"
