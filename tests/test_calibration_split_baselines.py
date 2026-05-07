import csv
import json

from scripts.run_baseline_analysis import bootstrap_baseline_metrics, run_baseline_analysis
from budget2success.baselines.calibration_split import (
    apply_histogram_recalibrator,
    fit_histogram_recalibrator,
    fit_source_by_budget,
    predict_source_by_budget,
)
from budget2success.schemas.records import TaskRecord


def test_source_prior_fallback_and_recalibrator_range():
    calibration_outcomes = [
        {"task_id": "cal", "budget": 64, "success": True, "metadata": {"source": "seen"}},
        {"task_id": "cal", "budget": 128, "success": False, "metadata": {"source": "seen"}},
    ]
    task_metadata = {"cal": TaskRecord(task_id="cal", track="math", prompt="p", verifier="numeric_exact", source="seen")}
    fitted = fit_source_by_budget(calibration_outcomes, task_metadata)
    predictions = predict_source_by_budget(["eval"], {"eval": TaskRecord(task_id="eval", track="math", prompt="p", verifier="numeric_exact", source="missing")}, {"eval": [64, 128]}, fitted)
    assert predictions["eval"][64] == fitted["__global__"][64]

    calibrator = fit_histogram_recalibrator(
        [{"task_id": "cal", "p_success_by_budget": {"64": 0.9, "128": 0.1}}],
        calibration_outcomes,
        n_bins=2,
    )
    curves = apply_histogram_recalibrator([{"task_id": "eval", "p_success_by_budget": {"64": 0.3}}], calibrator)
    assert 0 <= curves["eval"][64] <= 1


def test_run_baseline_analysis_uses_calibration_split(tmp_path):
    run_dir = tmp_path / "paper_math_core__m"
    run_dir.mkdir()
    forecasts = [
        {"task_id": "cal", "model": "m", "p_success_by_budget": {"64": 0.8, "128": 0.9}},
        {"task_id": "eval", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.4}},
    ]
    outcomes = [
        {"task_id": "cal", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval", "model": "m", "budget": 64, "success": False, "completion_tokens": 64, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval", "model": "m", "budget": 128, "success": True, "completion_tokens": 90, "metadata": {"track": "math", "source": "toy"}},
    ]
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "paper_math_core_calibration_eval_split.json").write_text(
        json.dumps(
            {
                "suite": "paper_math_core",
                "seed": 1,
                "calibration_frac": 0.5,
                "counts": {"calibration": 1, "evaluation": 1},
                "task_splits": {"cal": "calibration", "eval": "evaluation"},
            }
        ),
        encoding="utf-8",
    )

    table = tmp_path / "baseline_comparison.csv"
    run_baseline_analysis(
        suite="paper_math_core",
        run_dir=run_dir,
        output_table=table,
        output_summary=tmp_path / "baseline_summary.csv",
        output_figure_prefix=tmp_path / "fig",
        split_dir=split_dir,
        use_calibration_split=True,
        include_test_distribution_diagnostics=True,
        calibration_table=tmp_path / "paper_table4.csv",
        diagnostic_table=tmp_path / "paper_table5.csv",
        n_bootstrap=20,
    )

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert {row["baseline_class"] for row in rows} >= {
        "model_forecast_raw",
        "model_forecast_recalibrated",
        "calibration_split_baseline",
        "test_distribution_diagnostic",
        "posthoc_diagnostic",
    }
    assert all(row["n_eval_tasks"] == "1" for row in rows)
    assert all(row["calibration_task_ids_in_eval"] == "0" for row in rows)
    assert all("[" in row["brier_ci"] and "]" in row["brier_ci"] for row in rows)

    paper_rows = list(csv.DictReader((tmp_path / "paper_table4.csv").open(encoding="utf-8")))
    assert paper_rows
    assert all("[" in row["brier_ci"] and "]" in row["brier_ci"] for row in paper_rows)


def test_bootstrap_baseline_metrics_is_seed_deterministic():
    curves = {
        "method": {
            "t1": {64: 0.2, 128: 0.8},
            "t2": {64: 0.1, 128: 0.4},
            "t3": {64: 0.7, 128: 0.9},
        }
    }
    outcomes = {
        "t1": {64: False, 128: True},
        "t2": {64: False, 128: False},
        "t3": {64: True, 128: True},
    }

    first = bootstrap_baseline_metrics(curves, outcomes, n_bootstrap=25, seed=7)
    second = bootstrap_baseline_metrics(curves, outcomes, n_bootstrap=25, seed=7)

    assert first == second
    assert "brier" in first["method"]
    assert "[" in str(first["method"]["brier"]["ci"])
