import json

from scripts.make_paper_figures import make_paper_figures


def test_paper_figure_outputs_exist(tmp_path, monkeypatch):
    run_dir = tmp_path / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}, "median_budget2success": 128}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_id": "t1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}}),
                json.dumps({"task_id": "t1", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text("suite: paper_math_core\nmodel: m\nrun_id: m\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tables / "baseline_comparison.csv").write_text(
        "suite,run_id,model,forecast_method,baseline_class,track,source,n_tasks,n_eval_tasks,brier,ece,regret,notes\n"
        "paper_math_core,m,m,self_forecast_raw,model_forecast_raw,all,all,1,1,0.1,0.2,0.0,raw\n",
        encoding="utf-8",
    )
    (tables / "bootstrap_success_by_budget.csv").write_text(
        "suite,run_id,model,track,source,budget,success_rate,ci_low,ci_high\n"
        "paper_math_core,m,m,math,toy,64,0.0,0.0,0.2\n"
        "paper_math_core,m,m,math,toy,128,1.0,0.8,1.0\n",
        encoding="utf-8",
    )

    make_paper_figures(artifact_root=tmp_path / "artifacts", figure_dir=tmp_path / "reports" / "figures")

    required = [
        "paper_figure1_pipeline",
        "paper_figure2_success_by_budget",
        "paper_figure3_calibration_by_suite",
        "paper_figure4_budget_error_distribution",
        "paper_figure5_calibration_split_baselines",
        "paper_figure6_regret",
    ]
    for prefix in required:
        for suffix in [".png", ".svg"]:
            path = tmp_path / "reports" / "figures" / f"{prefix}{suffix}"
            assert path.exists()
            assert path.stat().st_size > 0
    qa_notes = (tmp_path / "reports" / "figures" / "paper_plot_qa_notes.md").read_text(encoding="utf-8")
    assert "Figure 2 CI data: used" in qa_notes
