from pathlib import Path


EXTENSION_TABLES = {
    "paper_table18_bigcodebench_hard.csv",
    "paper_table19_canitedit_descriptive.csv",
    "paper_table20_replacement_token_usage_proxy.csv",
    "paper_table21_replacement_allocation_frontier_raw.csv",
    "paper_table21_replacement_fixed_budget_scheduling.csv",
}


def test_no_mock_model_in_main_extension_tables():
    for name in EXTENSION_TABLES:
        path = Path("reports/tables") / name
        if not path.exists():
            continue
        assert "mock-model" not in path.read_text(encoding="utf-8")
