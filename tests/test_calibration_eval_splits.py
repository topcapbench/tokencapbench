import csv
import json

from scripts.build_calibration_eval_splits import build_calibration_eval_splits


def _write_run(root, suite="paper_math_core"):
    run_dir = root / f"{suite}__m"
    run_dir.mkdir(parents=True)
    forecasts = [
        {"task_id": f"t{i}", "model": "m", "p_success_by_budget": {"64": 0.2}, "metadata": {"source": "toy"}}
        for i in range(12)
    ]
    outcomes = [
        {"task_id": f"t{i}", "model": "m", "budget": 64, "success": i % 2 == 0, "metadata": {"source": "toy", "track": "math"}}
        for i in range(12)
    ]
    (run_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(f"suite: {suite}\nmodel: m\nrun_id: m\n", encoding="utf-8")


def test_calibration_eval_split_schema_and_determinism(tmp_path):
    artifact_root = tmp_path / "artifacts"
    _write_run(artifact_root)

    first = build_calibration_eval_splits(artifact_root=artifact_root, output_dir=tmp_path / "splits1", summary_path=tmp_path / "summary1.csv", calibration_frac=0.3, seed=123)
    second = build_calibration_eval_splits(artifact_root=artifact_root, output_dir=tmp_path / "splits2", summary_path=tmp_path / "summary2.csv", calibration_frac=0.3, seed=123)

    payload1 = json.loads(first[0].read_text(encoding="utf-8"))
    payload2 = json.loads(second[0].read_text(encoding="utf-8"))
    assert payload1 == payload2
    assert {"suite", "seed", "calibration_frac", "counts", "task_splits"} <= set(payload1)
    assert payload1["counts"]["calibration"] + payload1["counts"]["evaluation"] == 12
    assert set(payload1["task_splits"]) == {f"t{i}" for i in range(12)}

    with (tmp_path / "summary1.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["suite"] == "paper_math_core"
    assert int(rows[0]["n_tasks"]) == 12
