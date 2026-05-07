import csv
import json

from scripts.run_learned_calibration_baseline import (
    _calibration_eval_ids,
    learned_logistic_curves,
    run_learned_calibration_baseline,
)
from budget2success.schemas.records import TaskRecord


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_calibration_eval_ids_are_disjoint():
    outcomes = {"cal": {64: True}, "eval": {64: False}}

    calibration, evaluation = _calibration_eval_ids(
        {"cal": "calibration", "eval": "evaluation"},
        outcomes,
    )

    assert set(calibration).isdisjoint(evaluation)


def test_learned_baseline_emits_probabilities_in_range():
    tasks = {
        task.task_id: task
        for task in [
            TaskRecord(task_id="cal_low", track="math", prompt="short", verifier="numeric_exact", source="toy"),
            TaskRecord(task_id="cal_high", track="math", prompt="long prompt", verifier="numeric_exact", source="toy"),
            TaskRecord(task_id="eval_low", track="math", prompt="short eval", verifier="numeric_exact", source="toy"),
        ]
    }
    raw_curves = {
        "cal_low": {64: 0.1, 128: 0.2},
        "cal_high": {64: 0.8, 128: 0.9},
        "eval_low": {64: 0.2, 128: 0.3},
    }
    outcomes = {
        "cal_low": {64: False, 128: False},
        "cal_high": {64: True, 128: True},
        "eval_low": {64: False, 128: False},
    }

    curves = learned_logistic_curves(
        raw_curves=raw_curves,
        outcomes=outcomes,
        calibration_ids=["cal_low", "cal_high"],
        eval_ids=["eval_low"],
        task_records=tasks,
    )

    assert curves
    assert all(0.0 <= probability <= 1.0 for curve in curves.values() for probability in curve.values())


def test_script_writes_expected_columns_and_learned_improves_over_constant(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__m"
    artifact_dir.mkdir(parents=True)
    task_file = tmp_path / "tasks.jsonl"
    tasks = [
        TaskRecord(task_id="cal_low", track="math", prompt="short", verifier="numeric_exact", source="toy").model_dump(mode="json"),
        TaskRecord(task_id="cal_high", track="math", prompt="long prompt", verifier="numeric_exact", source="toy").model_dump(mode="json"),
        TaskRecord(task_id="eval_low", track="math", prompt="short eval", verifier="numeric_exact", source="toy").model_dump(mode="json"),
        TaskRecord(task_id="eval_high", track="math", prompt="long eval prompt", verifier="numeric_exact", source="toy").model_dump(mode="json"),
    ]
    _write_jsonl(task_file, tasks)
    forecasts = [
        {"task_id": "cal_low", "model": "m", "p_success_by_budget": {"64": 0.05, "128": 0.10}},
        {"task_id": "cal_high", "model": "m", "p_success_by_budget": {"64": 0.85, "128": 0.95}},
        {"task_id": "eval_low", "model": "m", "p_success_by_budget": {"64": 0.10, "128": 0.20}},
        {"task_id": "eval_high", "model": "m", "p_success_by_budget": {"64": 0.80, "128": 0.90}},
    ]
    outcomes = [
        {"task_id": "cal_low", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal_low", "model": "m", "budget": 128, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal_high", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal_high", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval_low", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval_low", "model": "m", "budget": 128, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval_high", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "eval_high", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
    ]
    _write_jsonl(artifact_dir / "forecasts.jsonl", forecasts)
    _write_jsonl(artifact_dir / "outcomes.jsonl", outcomes)
    (artifact_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "config_snapshot.yaml").write_text(
        f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "paper_math_core_calibration_eval_split.json").write_text(
        json.dumps(
            {
                "task_splits": {
                    "cal_low": "calibration",
                    "cal_high": "calibration",
                    "eval_low": "evaluation",
                    "eval_high": "evaluation",
                }
            }
        ),
        encoding="utf-8",
    )
    table = tmp_path / "paper_table16_learned_calibration_baseline.csv"

    run_learned_calibration_baseline(
        artifact_root=tmp_path / "artifacts",
        split_dir=split_dir,
        output_table=table,
        output_figure_prefix=tmp_path / "fig",
        n_bootstrap=0,
    )

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert rows
    assert set(rows[0]) == {
        "suite",
        "model",
        "method",
        "n_calibration_tasks",
        "n_eval_tasks",
        "brier",
        "ece",
        "regret",
        "brier_ci",
        "ece_ci",
        "regret_ci",
    }
    brier_by_method = {row["method"]: float(row["brier"]) for row in rows}
    assert brier_by_method["learned_logistic_recalibrator"] < brier_by_method["constant_by_budget_calibration"]
