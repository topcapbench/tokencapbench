import json

from budget2success.execution.aider_polyglot_bridge import AiderPolyglotBridge
from budget2success.schemas.records import TaskRecord


def test_aider_polyglot_verifier_runs_toy_pytest(tmp_path):
    exercise = tmp_path / "python" / "add_one"
    exercise.mkdir(parents=True)
    (exercise / "add_one.py").write_text("def add_one(x):\n    pass\n", encoding="utf-8")
    (exercise / "test_add_one.py").write_text("from add_one import add_one\n\ndef test_add_one():\n    assert add_one(1) == 2\n", encoding="utf-8")
    task = TaskRecord(
        task_id="aider_polyglot_python_add_one",
        track="coding_edit",
        source="aider_polyglot",
        prompt="Edit add_one.py.",
        verifier="aider_polyglot_tests",
        external_eval={
            "harness": "aider_polyglot_tests",
            "language": "python",
            "source_root": str(tmp_path),
            "exercise_dir": "python/add_one",
            "allowed_source_files": ["add_one.py"],
            "test_files": ["test_add_one.py"],
            "test_command": ["pytest", "-q"],
        },
    )
    solution = json.dumps({"files": [{"path": "add_one.py", "content": "def add_one(x):\n    return x + 1\n"}]})

    result = AiderPolyglotBridge().verify(task, solution)

    assert result.success
    assert result.metadata["label_source"] == "aider_polyglot_tests"
