import json

from scripts.validate_replacement_splits import validate_replacement_splits
from budget2success.schemas.records import TaskRecord


def _task(task_id, source, track, verifier, budget_grid, metadata):
    return TaskRecord(
        task_id=task_id,
        source=source,
        track=track,
        prompt="p",
        verifier=verifier,
        budget_grid=budget_grid,
        metadata=metadata,
        external_eval={"harness": verifier, "task_id": task_id},
    ).model_dump(mode="json")


def test_validate_replacement_splits_accepts_valid_small_override(tmp_path):
    canitedit = tmp_path / "canitedit.jsonl"
    bigcodebench = tmp_path / "bigcodebench.jsonl"
    canitedit.write_text(
        json.dumps(
            _task(
                "c1",
                "canitedit",
                "code_editing",
                "canitedit",
                [128],
                {"tests": "assert True", "chat_completion_compatible": True, "requires_docker": False},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    bigcodebench.write_text(
        json.dumps(
            _task(
                "b1",
                "bigcodebench_hard",
                "coding",
                "bigcodebench",
                [128],
                {"chat_completion_compatible": True, "requires_docker": False},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output, ok = validate_replacement_splits(
        canitedit=canitedit,
        bigcodebench=bigcodebench,
        output=tmp_path / "validation.csv",
        min_canitedit_tasks=1,
        expected_bigcodebench_tasks=1,
    )

    assert ok
    assert output.exists()
    assert "provided_test_coverage" in output.read_text(encoding="utf-8")
