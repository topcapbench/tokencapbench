import csv
import json

from scripts.analyze_token_usage_proxy import analyze_token_usage_proxy


def _write_run(run_dir, forecasts, outcomes):
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("suite: paper_math_core\nmodel: m\nrun_id: m\n", encoding="utf-8")


def test_token_usage_proxy_analysis_outputs_and_respects_split(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__m"
    forecasts = [
        {
            "task_id": "cal",
            "model": "m",
            "p_success_by_budget": {"64": 0.8, "128": 0.9},
            "forecast_extras": {"predicted_total_visible_tokens_to_solve": 64, "predicted_unconstrained_output_tokens": 80},
        },
        {
            "task_id": "eval1",
            "model": "m",
            "p_success_by_budget": {"64": 0.2, "128": 0.8},
            "forecast_extras": {"predicted_total_visible_tokens_to_solve": 120, "predicted_unconstrained_output_tokens": 140},
        },
        {
            "task_id": "eval2",
            "model": "m",
            "p_success_by_budget": {"64": 0.7, "128": 0.9},
            "forecast_extras": {"predicted_total_visible_tokens_to_solve": 64, "predicted_unconstrained_output_tokens": 70},
        },
    ]
    outcomes = [
        {"task_id": "cal", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval1", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval2", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval2", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
    ]
    _write_run(artifact_dir, forecasts, outcomes)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "paper_math_core_calibration_eval_split.json").write_text(
        json.dumps({"task_splits": {"cal": "calibration", "eval1": "evaluation", "eval2": "evaluation"}}),
        encoding="utf-8",
    )

    table = tmp_path / "paper_table13_token_usage_proxy.csv"
    figure_prefix = tmp_path / "paper_figure10_token_usage_proxy_vs_success"
    analyze_token_usage_proxy(
        artifact_root=tmp_path / "artifacts",
        split_dir=split_dir,
        dual_forecast_root=tmp_path / "missing_dual",
        output_table=table,
        output_figure_prefix=figure_prefix,
    )

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert rows
    row = rows[0]
    assert row["n_eval_tasks"] == "2"
    assert row["n_calibration_tasks"] == "1"
    assert row["calibration_task_ids_in_eval"] == "0"
    assert row["n_with_usage_forecast"] == "2"
    assert row["corr_predicted_total_visible_tokens_to_observed_first_success_budget"] != "NA"
    assert row["token_proxy_calibrated_brier"] != "NA"
    assert figure_prefix.with_suffix(".png").exists()
    assert figure_prefix.with_suffix(".png").stat().st_size > 0


def test_token_usage_proxy_analysis_handles_missing_extra_fields(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__m"
    forecasts = [
        {"task_id": "eval1", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}},
        {"task_id": "eval2", "model": "m", "p_success_by_budget": {"64": 0.7, "128": 0.9}},
    ]
    outcomes = [
        {"task_id": "eval1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval1", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval2", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval2", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
    ]
    _write_run(artifact_dir, forecasts, outcomes)

    table = tmp_path / "paper_table13_token_usage_proxy.csv"
    analyze_token_usage_proxy(
        artifact_root=tmp_path / "artifacts",
        split_dir=tmp_path / "missing_splits",
        dual_forecast_root=tmp_path / "missing_dual",
        output_table=table,
        output_figure_prefix=tmp_path / "fig",
    )

    row = list(csv.DictReader(table.open(encoding="utf-8")))[0]
    assert row["n_with_usage_forecast"] == "0"
    assert row["corr_predicted_total_visible_tokens_to_observed_first_success_budget"] == "NA"
    assert row["token_usage_proxy_ranking_accuracy"] == "NA"
    assert row["raw_success_brier"] != "NA"
