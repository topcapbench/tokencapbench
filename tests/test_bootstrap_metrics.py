import csv
import json

from scripts.bootstrap_metrics import bootstrap_suite


def test_bootstrap_metrics_outputs_stable_columns_and_shape(tmp_path):
    run_dir = tmp_path / "reports" / "runs" / "tiny"
    run_dir.mkdir(parents=True)
    forecasts = [
        {"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.8}, "median_budget2success": 128},
        {"task_id": "t2", "model": "m", "p_success_by_budget": {"64": 0.2, "128": 0.4}, "median_budget2success": 128},
    ]
    outcomes = [
        {"task_id": "t1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t1", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t2", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "t2", "model": "m", "budget": 128, "success": False, "metadata": {"track": "math", "source": "toy"}},
    ]
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    main_path, success_path = bootstrap_suite(run_dir=run_dir, n_bootstrap=25, seed=0, output_dir=tmp_path)

    with main_path.open(encoding="utf-8", newline="") as f:
        main_rows = list(csv.DictReader(f))
    with success_path.open(encoding="utf-8", newline="") as f:
        success_rows = list(csv.DictReader(f))
    assert {"metric", "estimate", "ci_low", "ci_high", "n_bootstrap"} <= set(main_rows[0])
    assert any(row["metric"] == "brier" for row in main_rows)
    assert any(row["metric"] == "truncation_rate" for row in main_rows)
    assert {row["budget"] for row in success_rows if row["source"] == "toy"} == {"64", "128"}
