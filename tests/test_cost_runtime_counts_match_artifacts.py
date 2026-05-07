import csv
from pathlib import Path

import pytest


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def test_cost_runtime_counts_match_artifacts():
    table_path = Path("reports/tables/paper_table6_cost_runtime.csv")
    assert table_path.exists()
    with table_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    data_rows = [row for row in rows if row.get("suite") and row.get("suite") != "TOTAL"]
    assert data_rows

    total_budgeted = 0
    total_artifact_rows = 0
    for row in data_rows:
        artifact_dir = Path("reports/artifacts") / f"{row['suite']}__{row['run_id']}"
        if not artifact_dir.exists():
            pytest.skip("raw report artifacts are not tracked in lightweight checkouts")
        assert artifact_dir.exists(), f"missing artifact directory for {row['suite']} {row['run_id']}"
        line_count = _line_count(artifact_dir / "outcomes.jsonl")
        assert int(row["budgeted_outcomes"]) == line_count
        assert int(row["artifact_outcome_rows"]) == line_count
        assert row["row_count_matches_artifact"].lower() == "true"
        total_budgeted += int(row["budgeted_outcomes"])
        total_artifact_rows += line_count

    total = next(row for row in rows if row.get("suite") == "TOTAL")
    assert int(total["budgeted_outcomes"]) == total_budgeted
    assert int(total["artifact_outcome_rows"]) == total_artifact_rows
    assert total["row_count_matches_artifact"].lower() == "true"
