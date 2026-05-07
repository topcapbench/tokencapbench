import csv
import json
from pathlib import Path


FORBIDDEN = {
    "official_labels_absent",
    "failed_or_incomplete",
    "swebench",
    "swe_verified",
    "bfcl",
    "aider_polyglot",
}


def test_no_non_provider_or_incomplete_rows_marked_main_in_paper_tables():
    for path in Path("reports/tables").glob("paper_table*.csv"):
        if not path.read_text(encoding="utf-8").strip():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            role = str(row.get("paper_role") or "").strip().lower()
            if role not in {"main", "paper", "main_text", "main_candidate"}:
                continue
            text = json.dumps(row, sort_keys=True).lower()
            assert not any(value in text for value in FORBIDDEN), f"{path} has forbidden main row: {row}"
