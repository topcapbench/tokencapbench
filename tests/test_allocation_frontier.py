import csv
import json

from scripts.run_allocation_frontier import (
    _select_at_or_below_target,
    _special_candidates,
    allocate_from_curves,
    run_allocation_frontier,
    summarize_fixed_budget_points,
)


def test_policy_allocation_has_at_most_one_budget_per_task():
    curves = {
        "t1": {64: 0.4, 128: 0.8},
        "t2": {64: 0.5, 128: 0.7},
    }
    outcomes = {
        "t1": {64: False, 128: True},
        "t2": {64: True, 128: True},
    }

    allocation = allocate_from_curves(curves, outcomes, 192, policy="policy_b")

    assert len(allocation) == len(set(allocation))
    assert set(allocation).issubset(outcomes)
    assert all(budget in outcomes[task_id] for task_id, budget in allocation.items())


def test_policy_b_does_not_undercount_nonmonotone_upgrade_cost():
    curves = {
        "t1": {64: 0.5, 128: 0.4, 192: 0.9},
    }
    outcomes = {
        "t1": {64: True, 128: True, 192: True},
    }

    allocation = allocate_from_curves(curves, outcomes, 128, policy="policy_b")

    assert sum(allocation.values()) <= 128
    assert allocation == {"t1": 64}


def test_random_baseline_is_seed_reproducible():
    outcomes = {
        "t1": {64: False, 128: True},
        "t2": {64: True, 128: True},
        "t3": {64: False, 128: False},
    }

    first = _special_candidates("random_budget", outcomes, seed=123)
    second = _special_candidates("random_budget", outcomes, seed=123)
    different = _special_candidates("random_budget", outcomes, seed=124)

    assert first == second
    assert first != different


def test_run_allocation_frontier_writes_outputs_and_oracle_dominates(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__m"
    artifact_dir.mkdir(parents=True)
    forecasts = [
        {"task_id": "cal", "model": "m", "p_success_by_budget": {"64": 0.7, "128": 0.9}},
        {"task_id": "e1", "model": "m", "p_success_by_budget": {"64": 0.3, "128": 0.8}},
        {"task_id": "e2", "model": "m", "p_success_by_budget": {"64": 0.6, "128": 0.7}},
    ]
    outcomes = [
        {"task_id": "cal", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "cal", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "e1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "e1", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "e2", "model": "m", "budget": 64, "success": True, "metadata": {"track": "math", "source": "toy"}},
        {"task_id": "e2", "model": "m", "budget": 128, "success": True, "metadata": {"track": "math", "source": "toy"}},
    ]
    (artifact_dir / "forecasts.jsonl").write_text("\n".join(json.dumps(row) for row in forecasts) + "\n", encoding="utf-8")
    (artifact_dir / "outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n", encoding="utf-8")
    (artifact_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "config_snapshot.yaml").write_text(
        "suite: paper_math_core\nmodel: m\nrun_id: m\n",
        encoding="utf-8",
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "paper_math_core_calibration_eval_split.json").write_text(
        json.dumps(
            {
                "suite": "paper_math_core",
                "task_splits": {"cal": "calibration", "e1": "evaluation", "e2": "evaluation"},
            }
        ),
        encoding="utf-8",
    )

    table = tmp_path / "paper_table12_allocation_frontier.csv"
    figures = tmp_path / "figures"
    run_allocation_frontier(
        artifact_root=tmp_path / "artifacts",
        split_dir=split_dir,
        output_table=table,
        figures_dir=figures,
        seed=99,
    )

    assert table.exists()
    assert table.stat().st_size > 0
    for name in [
        "paper_figure9_allocation_frontier.png",
        "paper_figure9_allocation_frontier.svg",
        "appendix_allocation_frontier_policy_a.png",
        "appendix_allocation_frontier_policy_a.svg",
    ]:
        path = figures / name
        assert path.exists()
        assert path.stat().st_size > 0

    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    assert rows
    assert {row["method"] for row in rows} >= {"oracle", "self_forecast_raw", "random_budget"}
    assert all(int(row["oracle_successes"]) >= int(row["verified_successes"]) for row in rows)
    assert all(int(row["regret_to_oracle"]) >= 0 for row in rows)


def test_select_at_or_below_target_never_overshoots():
    rows = [
        {"total_budget": 0, "budget_used": 0, "verified_successes": 0},
        {"total_budget": 100, "budget_used": 100, "verified_successes": 1},
        {"total_budget": 200, "budget_used": 200, "verified_successes": 3},
        {"total_budget": 400, "budget_used": 400, "verified_successes": 5},
    ]

    assert _select_at_or_below_target(rows, 250)["total_budget"] == 200
    assert _select_at_or_below_target(rows, 399)["total_budget"] == 200
    assert _select_at_or_below_target(rows, 400)["total_budget"] == 400
    assert _select_at_or_below_target(rows, 50)["total_budget"] == 0


def test_summarize_fixed_budget_points_uses_policy_b_and_strict_fraction():
    rows = [
        {
            "suite": "s",
            "model": "m",
            "method": "self_forecast_raw",
            "policy": "policy_a",
            "total_budget": 100,
            "verified_successes": 99,
        },
        {
            "suite": "s",
            "model": "m",
            "method": "self_forecast_raw",
            "policy": "policy_b",
            "total_budget": 0,
            "budget_used": 0,
            "allocated_tasks": 0,
            "verified_successes": 0,
            "success_rate": "0",
            "oracle_successes": 0,
            "regret_to_oracle": 0,
        },
        {
            "suite": "s",
            "model": "m",
            "method": "self_forecast_raw",
            "policy": "policy_b",
            "total_budget": 100,
            "budget_used": 96,
            "allocated_tasks": 2,
            "verified_successes": 1,
            "success_rate": "0.5",
            "oracle_successes": 2,
            "regret_to_oracle": 1,
        },
        {
            "suite": "s",
            "model": "m",
            "method": "self_forecast_raw",
            "policy": "policy_b",
            "total_budget": 200,
            "budget_used": 192,
            "allocated_tasks": 2,
            "verified_successes": 2,
            "success_rate": "1.0",
            "oracle_successes": 2,
            "regret_to_oracle": 0,
        },
    ]

    summary = summarize_fixed_budget_points(rows, budget_fractions=(0.5, 1.0))

    assert [row["selected_total_budget"] for row in summary] == [100, 200]
    assert [row["verified_successes"] for row in summary] == [1, 2]
    assert all(row["policy"] == "policy_b" for row in summary)
    assert [row["budget_slack_tokens"] for row in summary] == [4, 8]
    assert all(row["strict_budget_feasible"] == 1 for row in summary)
