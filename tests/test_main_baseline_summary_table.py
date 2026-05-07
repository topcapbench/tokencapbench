import csv

from scripts.make_paper_tables import make_paper_tables


def test_main_baseline_summary_table_is_compressed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tables / "baseline_comparison.csv").write_text(
        "suite,run_id,model,forecast_method,baseline_class,track,source,n_eval_tasks,brier,ece,regret,brier_ci,regret_ci,notes\n"
        "s,r,m,self_forecast_raw,model_forecast_raw,all,all,10,0.30,0.1,0.05,\"0.300 [0.2, 0.4]\",\"0.050 [0.0, 0.1]\",raw\n"
        "s,r,m,self_forecast_histogram_recalibrated,model_forecast_recalibrated,all,all,10,0.20,0.1,0.04,\"0.200 [0.1, 0.3]\",\"0.040 [0.0, 0.1]\",recal\n"
        "s,r,m,constant_by_budget_calibration,calibration_split_baseline,all,all,10,0.25,0.1,0.03,\"0.250 [0.2, 0.3]\",\"0.030 [0.0, 0.1]\",prior\n",
        encoding="utf-8",
    )

    make_paper_tables(artifact_root=tmp_path / "reports" / "artifacts", table_dir=tables)

    with (tables / "paper_table4_main_baseline_summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["raw_self_forecast_brier_ci"] == "0.300 [0.2, 0.4]"
    assert rows[0]["best_recalibrated_method"] == "self_forecast_histogram_recalibrated"
    assert rows[0]["best_simple_prior_method"] == "constant_by_budget_calibration"
    assert rows[0]["best_regret_method"] == "constant_by_budget_calibration"
