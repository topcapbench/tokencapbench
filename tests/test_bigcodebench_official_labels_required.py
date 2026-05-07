import csv
from pathlib import Path


def test_bigcodebench_official_labels_required_for_main():
    path = Path("reports/tables/paper_table18_bigcodebench_hard.csv")
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("official_harness_status") == "official_labels_absent":
            assert row.get("paper_role", "appendix") not in {"main", "paper", "main_text", "main_candidate"}
