import json

from budget2success.datasets.base import AdapterConfig
from budget2success.datasets.bigcodebench_hard import BigCodeBenchHardAdapter


def test_bigcodebench_hard_loader_uses_local_hard_rows(tmp_path):
    export = tmp_path / "bcb.jsonl"
    export.write_text(
        json.dumps(
            {
                "task_id": "1",
                "difficulty": "hard",
                "instruct_prompt": "Write add_one.",
                "function_signature": "def add_one(x):",
                "libs": ["math"],
            }
        )
        + "\n"
        + json.dumps({"task_id": "2", "difficulty": "easy", "instruct_prompt": "Write noop."})
        + "\n",
        encoding="utf-8",
    )
    adapter = BigCodeBenchHardAdapter(
        AdapterConfig(name="bigcodebench_hard", limit=148, budget_grid=[256], kwargs={"path": str(export)})
    )

    tasks = adapter.load_tasks()

    assert len(tasks) == 1
    assert tasks[0].task_id == "bigcodebench_hard_1"
    assert tasks[0].source == "bigcodebench_hard"
    assert tasks[0].track == "coding"
    assert tasks[0].verifier == "bigcodebench"
    assert tasks[0].budget_grid == [256]
    assert tasks[0].external_eval["harness"] == "bigcodebench"
    assert tasks[0].metadata["chat_completion_compatible"] is True
    assert tasks[0].metadata["requires_docker"] is False
    assert "def add_one" in tasks[0].prompt
