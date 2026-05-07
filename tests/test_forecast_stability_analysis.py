import csv
import json

from scripts.analyze_forecast_stability import analyze_forecast_stability


def test_forecast_stability_outputs_repeat_metrics(tmp_path):
    for index, probabilities in enumerate([{"64": 0.2, "128": 0.7}, {"64": 0.3, "128": 0.6}], start=1):
        run_dir = tmp_path / "artifacts" / f"paper_repeatability_small__m__forecast_repeat_{index}__solver_repeat_1"
        run_dir.mkdir(parents=True)
        forecasts = [
            {"task_id": "t1", "model": "m", "p_success_by_budget": probabilities},
            {"task_id": "t2", "model": "m", "p_success_by_budget": {"64": 0.8, "128": 0.9}},
        ]
        outcomes = [
            {"task_id": "t1", "model": "m", "budget": 64, "success": False},
            {"task_id": "t1", "model": "m", "budget": 128, "success": True},
            {"task_id": "t2", "model": "m", "budget": 64, "success": True},
            {"task_id": "t2", "model": "m", "budget": 128, "success": True},
        ]
        (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
        (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "config_snapshot.yaml").write_text(
            "suite: paper_repeatability_small\nmodel: m\nrun_id: m\nforecast_prompt: prompts/forecast_prompt.md\n",
            encoding="utf-8",
        )

    table = tmp_path / "paper_table14_forecast_stability.csv"
    figure_prefix = tmp_path / "appendix_forecast_stability"
    analyze_forecast_stability(
        artifact_root=tmp_path / "artifacts",
        output_table=table,
        output_figure_prefix=figure_prefix,
    )

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert rows
    assert rows[0]["forecast_groups"] == "2"
    assert rows[0]["tasks_with_repeats"] == "2"
    assert rows[0]["mean_probability_std"] != "NA"
    assert rows[0]["selected_budget_agreement"] != "NA"
    assert figure_prefix.with_suffix(".png").exists()
    assert figure_prefix.with_suffix(".png").stat().st_size > 0


def test_forecast_stability_loads_live_prompt_variants_with_reused_outcomes(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__m"
    artifact_dir.mkdir(parents=True)
    outcomes = [
        {"task_id": "t1", "model": "m", "budget": 64, "success": False},
        {"task_id": "t1", "model": "m", "budget": 128, "success": True},
    ]
    (artifact_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (artifact_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}}) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "config_snapshot.yaml").write_text("suite_name: paper_math_core\nmodel: m\nrun_id: m\n", encoding="utf-8")

    live_root = tmp_path / "runs" / "paper_forecast_stability"
    for name, prompt, probability in [("m__prompt_a__forecast_repeat_1__solver_repeat_1", "a.md", 0.25), ("m__prompt_b__forecast_repeat_1__solver_repeat_1", "b.md", 0.35)]:
        run_dir = live_root / name
        run_dir.mkdir(parents=True)
        (run_dir / "forecasts.jsonl").write_text(
            json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": probability, "128": 0.8}}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "config_snapshot.yaml").write_text(
            f"suite_name: paper_forecast_stability\nmodel: m\nrun_id: {name}\nforecast_prompt: {prompt}\n",
            encoding="utf-8",
        )

    table = tmp_path / "paper_table14_forecast_stability.csv"
    analyze_forecast_stability(
        artifact_root=tmp_path / "artifacts",
        live_run_root=live_root,
        output_table=table,
        output_figure_prefix=tmp_path / "appendix_forecast_stability",
    )

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    live_row = next(row for row in rows if row["suite"] == "paper_forecast_stability")
    assert live_row["forecast_groups"] == "2"
    assert live_row["prompt_variants_observed"] == "2"
    assert live_row["brier_std"] != "NA"
