import csv
import json
from pathlib import Path

from scripts.summarize_resource_metrics import summarize


def test_summarize_resource_metrics_groups_budget_rows(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    rows = [
        {
            "task_id": "a",
            "model": "m",
            "budget": 64,
            "success": True,
            "truncated": False,
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "generation_wall_time_seconds": 0.2,
            "verification_wall_time_seconds": 0.1,
            "end_to_end_wall_time_seconds": 0.3,
            "metadata": {"source": "toy"},
        },
        {
            "task_id": "b",
            "model": "m",
            "budget": 64,
            "success": False,
            "truncated": True,
            "finish_reason": "length",
            "prompt_tokens": 12,
            "completion_tokens": 64,
            "total_tokens": 76,
            "generation_wall_time_seconds": 0.4,
            "verification_wall_time_seconds": 0.1,
            "end_to_end_wall_time_seconds": 0.5,
            "metadata": {"source": "toy"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = summarize([path])
    assert len(summary) == 1
    row = summary[0]
    assert row["success_rate"] == 0.5
    assert row["truncated_rate"] == 0.5
    assert row["completion_tokens_median"] == 42.0
    assert row["end_to_end_wall_time_seconds_median"] == 0.4
