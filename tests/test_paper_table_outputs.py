import csv
import json
from pathlib import Path

from scripts.make_paper_tables import make_paper_tables


STRICT_FIXED_BUDGET_TABLES = [
    Path("reports/tables/paper_table15_fixed_budget_scheduling.csv"),
    Path("reports/tables/paper_table21_replacement_fixed_budget_scheduling.csv"),
]


def _write_run(root):
    run_dir = root / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.8}, "median_budget2success": 64}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("suite: paper_math_core\nmodel: m\nrun_id: m\n", encoding="utf-8")


def test_paper_table_outputs_and_columns(tmp_path, monkeypatch):
    _write_run(tmp_path / "artifacts")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (tmp_path / "reports" / "tables" / "baseline_comparison.csv").write_text(
        "suite,run_id,model,forecast_method,baseline_class,track,source,n_tasks,n_eval_tasks,brier,ece,regret,notes\n"
        "paper_math_core,m,m,self_forecast_raw,model_forecast_raw,all,all,1,1,0.1,0.2,0.0,raw\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "tables" / "math_verifier_delta_summary.csv").write_text(
        "suite,run_id,model,source,n_rows,n_changed,change_rate,old_success_rate,new_success_rate,success_delta,verifier_mode\n"
        "paper_math_core,m,m,toy,1,0,0,1,1,0,strict\n",
        encoding="utf-8",
    )

    make_paper_tables(artifact_root=tmp_path / "artifacts", table_dir=tmp_path / "reports" / "tables")

    expected = [
        "paper_table1_related_work.csv",
        "paper_table2_dataset_composition.csv",
        "paper_table_clean_evidence_scope.csv",
        "paper_table3a_calibration_capability.csv",
        "paper_table3b_allocation_diagnostics.csv",
        "paper_table4_calibration_split_baselines.csv",
        "paper_table5_diagnostic_baselines.csv",
        "paper_table6_cost_runtime.csv",
        "paper_table7_verifier_robustness.csv",
        "paper_table8_metric_definitions.csv",
        "paper_table9_release_checklist.csv",
    ]
    for name in expected:
        assert (tmp_path / "reports" / "tables" / name).exists()

    with (tmp_path / "reports" / "tables" / "paper_table_clean_evidence_scope.csv").open(encoding="utf-8", newline="") as f:
        clean_rows = list(csv.DictReader(f))
    assert [row["source"] for row in clean_rows] == [
        "GSM8K + MATH",
        "HumanEval+ + MBPP+",
        "BigCodeBench-Hard",
        "CanItEdit",
        "LiveCodeBench-300",
        "prompt variants",
    ]

    with (tmp_path / "reports" / "tables" / "paper_table3a_calibration_capability.csv").open(encoding="utf-8", newline="") as f:
        columns = set(next(csv.DictReader(f)).keys())
    assert {"suite", "model", "brier_ci", "ranking_accuracy_ci", "estimated_cost_usd"} <= columns

    with (tmp_path / "reports" / "tables" / "paper_table3b_allocation_diagnostics.csv").open(encoding="utf-8", newline="") as f:
        table3b_columns = set(next(csv.DictReader(f)).keys())
    assert "normalized_regret_ci" in table3b_columns

    with (tmp_path / "reports" / "tables" / "paper_table6_cost_runtime.csv").open(encoding="utf-8", newline="") as f:
        cost_rows = list(csv.DictReader(f))
    assert cost_rows[-1]["suite"] == "TOTAL"

    with (tmp_path / "reports" / "tables" / "paper_table7_verifier_robustness.csv").open(encoding="utf-8", newline="") as f:
        verifier_columns = set(next(csv.DictReader(f)).keys())
    assert {"verifier_policy", "math_verify_available", "unsupported_rows"} <= verifier_columns

    _assert_strict_fixed_budget_invariants(tmp_path / "reports" / "tables" / "paper_table15_fixed_budget_scheduling.csv")
    _assert_strict_fixed_budget_invariants(tmp_path / "reports" / "tables" / "paper_table21_replacement_fixed_budget_scheduling.csv")


def test_regenerated_fixed_budget_tables_obey_strict_invariants():
    for path in STRICT_FIXED_BUDGET_TABLES:
        if not path.exists():
            continue
        _assert_strict_fixed_budget_invariants(path)


def _assert_strict_fixed_budget_invariants(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    required = {
        "target_total_budget",
        "selected_total_budget",
        "budget_used",
        "budget_slack_tokens",
        "strict_budget_feasible",
    }
    assert required <= columns
    for row in rows:
        target_total_budget = int(float(row["target_total_budget"]))
        selected_total_budget = int(float(row["selected_total_budget"]))
        budget_used = int(float(row["budget_used"]))
        budget_slack_tokens = int(float(row["budget_slack_tokens"]))
        assert selected_total_budget <= target_total_budget
        assert budget_used <= selected_total_budget
        assert budget_used <= target_total_budget
        assert budget_slack_tokens == target_total_budget - budget_used
        assert budget_slack_tokens >= 0
        assert row["strict_budget_feasible"] in {"1", "true", "True", "yes"}
