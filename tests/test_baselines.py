import csv
import json

from scripts.run_baseline_analysis import run_baseline_analysis


def test_baseline_analysis_writes_classes_and_summary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    forecasts = [
        {"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}},
        {"task_id": "t2", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.4}},
    ]
    outcomes = [
        {"task_id": "t1", "model": "m", "budget": 64, "success": False, "completion_tokens": 30, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t1", "model": "m", "budget": 128, "success": True, "completion_tokens": 70, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t2", "model": "m", "budget": 64, "success": False, "completion_tokens": 64, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t2", "model": "m", "budget": 128, "success": False, "completion_tokens": 80, "metadata": {"track": "math", "source": "toy"}},
    ]
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    table = tmp_path / "baseline_comparison.csv"
    summary = tmp_path / "baseline_summary.csv"
    run_baseline_analysis(run_dir=run_dir, output_table=table, output_summary=summary, output_figure_prefix=tmp_path / "fig")

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    methods = {row["forecast_method"] for row in rows}
    assert {"constant_empirical_prior", "source_empirical_prior", "prompt_length_empirical"} <= methods
    assert any(row["forecast_method"] == "output_length_proxy" and row["baseline_class"] == "posthoc_diagnostic" for row in rows)
    assert summary.exists()


def test_calibration_split_bootstrap_table_is_deterministic_and_separates_diagnostics(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    task_file = tmp_path / "tasks.jsonl"
    tasks = [
        {"task_id": f"t{i}", "track": "math", "prompt": f"Compute {i}.", "verifier": "numeric_exact", "answer": str(i), "source": "gsm8k", "budget_grid": [64, 128]}
        for i in range(4)
    ]
    task_file.write_text("\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8")
    forecasts = [
        {"task_id": "t0", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}},
        {"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.3, "128": 0.7}},
        {"task_id": "t2", "model": "m", "p_success_by_budget": {"64": 0.4, "128": 0.9}},
        {"task_id": "t3", "model": "m", "p_success_by_budget": {"64": 0.5, "128": 0.6}},
    ]
    outcomes = []
    for task_id in ["t0", "t1", "t2", "t3"]:
        outcomes.append({"task_id": task_id, "model": "m", "budget": 64, "success": task_id in {"t0", "t2"}, "completion_tokens": 20, "metadata": {"track": "math", "source": "gsm8k"}})
        outcomes.append({"task_id": task_id, "model": "m", "budget": 128, "success": task_id != "t3", "completion_tokens": 40, "metadata": {"track": "math", "source": "gsm8k"}})
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n", encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "paper_math_core_calibration_eval_split.json").write_text(
        json.dumps({"task_splits": {"t0": "calibration", "t1": "calibration", "t2": "evaluation", "t3": "evaluation"}}),
        encoding="utf-8",
    )

    calibration_table = tmp_path / "calibration.csv"
    diagnostic_table = tmp_path / "diagnostic.csv"
    run_baseline_analysis(
        artifact_root=artifact_root,
        split_dir=split_dir,
        use_calibration_split=True,
        include_test_distribution_diagnostics=True,
        bootstrap=True,
        n_bootstrap=5,
        bootstrap_seed=123,
        output_table=tmp_path / "baseline.csv",
        output_summary=tmp_path / "summary.csv",
        output_figure_prefix=tmp_path / "fig",
        calibration_table=calibration_table,
        diagnostic_table=diagnostic_table,
    )

    first = calibration_table.read_text(encoding="utf-8")
    assert "[point]" not in first
    assert "brier_low" in first
    assert "output_length_proxy_posthoc" not in first
    assert "output_length_proxy_posthoc" in diagnostic_table.read_text(encoding="utf-8")

    run_baseline_analysis(
        artifact_root=artifact_root,
        split_dir=split_dir,
        use_calibration_split=True,
        include_test_distribution_diagnostics=True,
        bootstrap=True,
        n_bootstrap=5,
        bootstrap_seed=123,
        output_table=tmp_path / "baseline2.csv",
        output_summary=tmp_path / "summary2.csv",
        output_figure_prefix=tmp_path / "fig2",
        calibration_table=tmp_path / "calibration2.csv",
        diagnostic_table=tmp_path / "diagnostic2.csv",
    )
    assert first == (tmp_path / "calibration2.csv").read_text(encoding="utf-8")
