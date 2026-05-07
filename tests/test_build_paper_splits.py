from pathlib import Path

from scripts.build_paper_splits import build_splits
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


def test_build_paper_splits_small_records_validate(tmp_path):
    counts = build_splits(small=True, output_dir=tmp_path)

    assert counts
    for path_text, count in counts.items():
        path = Path(path_text)
        assert path.exists()
        assert count >= 1
        for row in read_jsonl(path):
            task = TaskRecord.model_validate(row)
            assert task.budget_grid
            assert row.get("fresh_split") is not None or "fresh_split" not in row
            assert row.get("verifier_policy") is not None or "verifier_policy" not in row
