#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import forecast_curves, load_paper_runs, outcomes_by_task, task_metadata
from budget2success.baselines.calibration_split import (
    apply_histogram_recalibrator,
    fit_constant_by_budget,
    fit_histogram_recalibrator,
    fit_prompt_length_bins,
    fit_source_by_budget,
    predict_constant_by_budget,
    predict_prompt_length_bins,
    predict_source_by_budget,
)
from budget2success.baselines.single_budget import single_budget_curve
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


METHODS = (
    "self_forecast_raw",
    "self_forecast_histogram_recalibrated",
    "constant_by_budget_calibration",
    "source_by_budget_calibration",
    "prompt_length_bin_calibration",
    "single_budget_midpoint",
    "random_budget",
    "cheapest_budget",
    "max_budget",
    "oracle",
)
SIMPLE_PRIORS = {
    "constant_by_budget_calibration",
    "source_by_budget_calibration",
    "prompt_length_bin_calibration",
    "single_budget_midpoint",
}
FIXED_BUDGET_SIMPLE_PRIORS = (
    "constant_by_budget_calibration",
    "source_by_budget_calibration",
    "prompt_length_bin_calibration",
)
FIXED_BUDGET_METHODS = (
    "self_forecast_raw",
    "self_forecast_histogram_recalibrated",
    "best_simple_prior",
    "random_budget",
    "max_budget",
    "oracle",
)
TABLE_FIELDS = [
    "suite",
    "model",
    "method",
    "policy",
    "total_budget",
    "allocated_tasks",
    "verified_successes",
    "success_rate",
    "budget_used",
    "oracle_successes",
    "regret_to_oracle",
]
FIXED_BUDGET_FIELDS = [
    "suite",
    "model",
    "method",
    "source_method",
    "policy",
    "budget_fraction",
    "target_total_budget",
    "selected_total_budget",
    "budget_used",
    "budget_slack_tokens",
    "strict_budget_feasible",
    "allocated_tasks",
    "verified_successes",
    "success_rate",
    "oracle_successes",
    "regret_to_oracle",
]
REPLACEMENT_FIXED_BUDGET_FIELDS = [
    "suite",
    "model",
    "policy",
    "global_budget_fraction",
    "target_total_budget",
    "selected_total_budget",
    "total_budget",
    "verified_successes",
    "oracle_successes",
    "regret_to_oracle",
    "budget_used",
    "budget_slack_tokens",
    "strict_budget_feasible",
]
REPLACEMENT_FIXED_METHODS = (
    "random_budget",
    "constant_by_budget_calibration",
    "source_by_budget_calibration",
    "self_forecast_raw",
    "self_forecast_histogram_recalibrated",
    "oracle",
)


def run_allocation_frontier(
    *,
    artifact_root: str | Path | list[str | Path] | tuple[str | Path, ...] = "reports/artifacts",
    split_dir: str | Path = "reports/splits",
    output_table: str | Path = "reports/tables/paper_table12_allocation_frontier.csv",
    figures_dir: str | Path = "reports/figures",
    seed: int = 20260501,
    write_figures: bool = True,
    fixed_budget_table: str | Path | None = None,
    suite_filter: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    allowed_suites = {str(value) for value in suite_filter} if suite_filter else None
    if allowed_suites:
        runs = []
        for root in _artifact_roots(artifact_root):
            for suite_name in sorted(allowed_suites):
                runs.extend(
                    load_paper_runs(
                        suite=suite_name,
                        run_root="reports/runs",
                        artifact_root=root,
                        include_artifacts=True,
                    )
                )
    else:
        runs = []
        for root in _artifact_roots(artifact_root):
            runs.extend(
                load_paper_runs(
                    run_root=Path(root) / "__no_reports_runs__",
                    artifact_root=root,
                    include_artifacts=True,
                )
            )
    runs = [run for run in runs if run.model != "mock-model"]
    for run in runs:
        run_suite = run.suite or ""
        if allowed_suites is not None and run_suite not in allowed_suites:
            continue
        outcomes = outcomes_by_task(run.outcomes)
        split_map = _load_split_map(Path(split_dir), run_suite)
        eval_task_ids = sorted(
            task_id
            for task_id, split in split_map.items()
            if split == "evaluation" and task_id in outcomes
        )
        if not eval_task_ids:
            eval_task_ids = sorted(outcomes)
        eval_outcomes = {task_id: outcomes[task_id] for task_id in eval_task_ids if outcomes.get(task_id)}
        if not eval_outcomes:
            continue
        capacities = _frontier_capacities(eval_outcomes)
        oracle_allocations = {capacity: allocate_oracle(eval_outcomes, capacity) for capacity in capacities}
        oracle_successes = {
            capacity: _verified_successes(oracle_allocations[capacity], eval_outcomes) for capacity in capacities
        }
        curves_by_method = _curves_by_method(run, eval_task_ids, eval_outcomes, split_map=split_map)
        special_orders = {
            method: _special_candidates(method, eval_outcomes, seed=_stable_seed(seed, run_suite, run.model, method))
            for method in {"random_budget", "cheapest_budget", "max_budget"}
        }
        for policy in ("policy_b", "policy_a"):
            for method in METHODS:
                for capacity in capacities:
                    if method == "oracle":
                        allocation = oracle_allocations[capacity]
                    elif method in special_orders:
                        allocation = allocate_special(special_orders[method], capacity)
                    else:
                        curves = curves_by_method.get(method, {})
                        allocation = allocate_from_curves(curves, eval_outcomes, capacity, policy=policy)
                    successes = _verified_successes(allocation, eval_outcomes)
                    allocated_tasks = len(allocation)
                    rows.append(
                        {
                            "suite": run_suite,
                            "model": run.model,
                            "method": method,
                            "policy": policy,
                            "total_budget": int(capacity),
                            "allocated_tasks": allocated_tasks,
                            "verified_successes": int(successes),
                            "success_rate": _format_float(successes / allocated_tasks if allocated_tasks else 0.0),
                            "budget_used": int(sum(allocation.values())),
                            "oracle_successes": int(oracle_successes[capacity]),
                            "regret_to_oracle": int(oracle_successes[capacity] - successes),
                        }
                    )

    table_path = Path(output_table)
    _write_csv(table_path, rows)
    outputs = [table_path]
    if write_figures:
        outputs.extend(plot_allocation_frontier(table_path=table_path, figures_dir=figures_dir))
    if fixed_budget_table is not None:
        outputs.extend(
            write_fixed_budget_scheduling_table(
                frontier_table=table_path,
                output_table=fixed_budget_table,
                figures_dir=figures_dir,
            )
        )
    return outputs


def allocate_from_curves(
    curves: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
    capacity: int,
    *,
    policy: str,
) -> dict[str, int]:
    filtered = _filter_curves_to_outcomes(curves, outcomes)
    if policy == "policy_a":
        return _allocate_policy_a(filtered, capacity)
    if policy == "policy_b":
        return _allocate_policy_b(filtered, capacity)
    raise ValueError(f"Unknown allocation policy: {policy}")


def allocate_special(candidates: list[tuple[str, int]], capacity: int) -> dict[str, int]:
    allocation: dict[str, int] = {}
    used = 0
    for task_id, budget in candidates:
        if task_id in allocation:
            continue
        if used + budget > capacity:
            continue
        allocation[task_id] = int(budget)
        used += int(budget)
    return allocation


def allocate_oracle(outcomes: dict[str, dict[int, bool]], capacity: int) -> dict[str, int]:
    """Exact success-count oracle for unit-value tasks.

    A task is worth one success at most, so the optimal upper bound under a
    token budget is to buy the cheapest verified-success budget for as many
    tasks as possible. Failed budgets never improve the oracle objective.
    """

    candidates: list[tuple[int, str]] = []
    for task_id, task_outcomes in outcomes.items():
        successful_budgets = [int(budget) for budget, success in task_outcomes.items() if success]
        if successful_budgets:
            candidates.append((min(successful_budgets), task_id))
    allocation: dict[str, int] = {}
    used = 0
    for budget, task_id in sorted(candidates):
        if used + budget > capacity:
            continue
        allocation[task_id] = budget
        used += budget
    return allocation


def plot_allocation_frontier(
    *,
    table_path: str | Path = "reports/tables/paper_table12_allocation_frontier.csv",
    figures_dir: str | Path = "reports/figures",
) -> list[Path]:
    rows = _read_csv(Path(table_path))
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(_plot_policy(rows, "policy_b", figures_dir / "paper_figure9_allocation_frontier"))
    outputs.extend(_plot_policy(rows, "policy_a", figures_dir / "appendix_allocation_frontier_policy_a"))
    return outputs


def summarize_fixed_budget_points(
    rows: list[dict[str, Any]],
    *,
    budget_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> list[dict[str, Any]]:
    """Summarize verified successes at fixed fractions of max global budget."""

    policy_rows = [row for row in rows if str(row.get("policy") or "") == "policy_b"]
    best_simple_by_group = _best_simple_prior_by_suite_model(policy_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        suite = str(row.get("suite") or "")
        model = str(row.get("model") or "")
        method = str(row.get("method") or "")
        source_method = method
        if method in FIXED_BUDGET_SIMPLE_PRIORS:
            selected = best_simple_by_group.get((suite, model))
            if method != selected:
                continue
            method = "best_simple_prior"
        elif method not in FIXED_BUDGET_METHODS:
            continue
        grouped[(suite, model, method)].append({**row, "source_method": source_method})

    summary_rows: list[dict[str, Any]] = []
    for (suite, model, method), group_rows in sorted(grouped.items()):
        if not group_rows:
            continue
        max_budget = max(_int(row.get("total_budget")) for row in group_rows)
        if max_budget <= 0:
            continue
        for fraction in budget_fractions:
            target_budget = int(round(max_budget * float(fraction)))
            selected = _select_at_or_below_target(group_rows, target_budget)
            selected_total_budget = _as_int(selected.get("total_budget"))
            budget_used = _as_int(selected.get("budget_used"))
            budget_slack_tokens = target_budget - budget_used
            strict_budget_feasible = _strict_budget_feasible(
                target_total_budget=target_budget,
                selected_total_budget=selected_total_budget,
                budget_used=budget_used,
                budget_slack_tokens=budget_slack_tokens,
            )
            if not strict_budget_feasible:
                raise ValueError(
                    "Strict fixed-budget selection produced an infeasible row: "
                    f"suite={suite} model={model} method={method} "
                    f"target_total_budget={target_budget} "
                    f"selected_total_budget={selected_total_budget} budget_used={budget_used}"
                )
            summary_rows.append(
                {
                    "suite": suite,
                    "model": model,
                    "method": method,
                    "source_method": selected.get("source_method", method),
                    "policy": "policy_b",
                    "budget_fraction": f"{float(fraction):.2f}",
                    "target_total_budget": target_budget,
                    "selected_total_budget": selected_total_budget,
                    "budget_used": budget_used,
                    "budget_slack_tokens": budget_slack_tokens,
                    "strict_budget_feasible": int(strict_budget_feasible),
                    "allocated_tasks": _as_int(selected.get("allocated_tasks")),
                    "verified_successes": _as_int(selected.get("verified_successes")),
                    "success_rate": selected.get("success_rate", ""),
                    "oracle_successes": _as_int(selected.get("oracle_successes")),
                    "regret_to_oracle": _as_int(selected.get("regret_to_oracle")),
                }
            )
    return summary_rows


def write_fixed_budget_scheduling_table(
    *,
    frontier_table: str | Path = "reports/tables/paper_table12_allocation_frontier.csv",
    output_table: str | Path = "reports/tables/paper_table15_fixed_budget_scheduling.csv",
    figures_dir: str | Path = "reports/figures",
) -> list[Path]:
    rows = _read_csv(Path(frontier_table))
    fixed_rows = summarize_fixed_budget_points(rows)
    output_path = Path(output_table)
    _write_csv_with_fields(output_path, fixed_rows, FIXED_BUDGET_FIELDS)
    outputs = [output_path]
    outputs.extend(plot_fixed_budget_scheduling(table_path=output_path, figures_dir=figures_dir))
    return outputs


def write_replacement_fixed_budget_scheduling_table(
    *,
    frontier_table: str | Path = "reports/tables/paper_table21_replacement_allocation_frontier_raw.csv",
    output_table: str | Path = "reports/tables/paper_table21_replacement_fixed_budget_scheduling.csv",
    figures_dir: str | Path = "reports/figures",
) -> list[Path]:
    rows = _read_csv(Path(frontier_table))
    fixed_rows = summarize_replacement_fixed_budget_points(rows)
    output_path = Path(output_table)
    _write_csv_with_fields(output_path, fixed_rows, REPLACEMENT_FIXED_BUDGET_FIELDS)
    outputs = [output_path]
    outputs.extend(plot_replacement_allocation_frontier(table_path=output_path, figures_dir=figures_dir))
    return outputs


def summarize_replacement_fixed_budget_points(
    rows: list[dict[str, Any]],
    *,
    budget_fractions: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> list[dict[str, Any]]:
    policy_rows = [
        row
        for row in rows
        if str(row.get("policy") or "") == "policy_b" and str(row.get("method") or "") in REPLACEMENT_FIXED_METHODS
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped[(str(row.get("suite") or ""), str(row.get("model") or ""), str(row.get("method") or ""))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (suite, model, method), group_rows in sorted(grouped.items()):
        if not group_rows:
            continue
        max_budget = max(_int(row.get("total_budget")) for row in group_rows)
        if max_budget <= 0:
            continue
        for fraction in budget_fractions:
            target_budget = int(round(max_budget * float(fraction)))
            selected = _select_at_or_below_target(group_rows, target_budget)
            selected_total_budget = _as_int(selected.get("total_budget"))
            budget_used = _as_int(selected.get("budget_used"))
            budget_slack_tokens = target_budget - budget_used
            strict_budget_feasible = _strict_budget_feasible(
                target_total_budget=target_budget,
                selected_total_budget=selected_total_budget,
                budget_used=budget_used,
                budget_slack_tokens=budget_slack_tokens,
            )
            if not strict_budget_feasible:
                raise ValueError(
                    "Strict replacement fixed-budget selection produced an infeasible row: "
                    f"suite={suite} model={model} method={method} "
                    f"target_total_budget={target_budget} "
                    f"selected_total_budget={selected_total_budget} budget_used={budget_used}"
                )
            summary_rows.append(
                {
                    "suite": suite,
                    "model": model,
                    "policy": _replacement_policy_label(method),
                    "global_budget_fraction": f"{fraction:.2f}",
                    "target_total_budget": target_budget,
                    "selected_total_budget": selected_total_budget,
                    "total_budget": selected_total_budget,
                    "verified_successes": _as_int(selected.get("verified_successes")),
                    "oracle_successes": _as_int(selected.get("oracle_successes")),
                    "regret_to_oracle": _as_int(selected.get("regret_to_oracle")),
                    "budget_used": budget_used,
                    "budget_slack_tokens": budget_slack_tokens,
                    "strict_budget_feasible": int(strict_budget_feasible),
                }
            )
    return summary_rows


def plot_replacement_allocation_frontier(
    *,
    table_path: str | Path = "reports/tables/paper_table21_replacement_fixed_budget_scheduling.csv",
    figures_dir: str | Path = "reports/figures",
    output_prefix: str | Path | None = None,
) -> list[Path]:
    rows = _read_csv(Path(table_path))
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(output_prefix) if output_prefix is not None else figures_dir / "paper_figure14_replacement_allocation_frontier"
    if not rows:
        return _save_empty(prefix, "No replacement allocation rows")
    suites = sorted({row["suite"] for row in rows if row.get("suite")})
    ncols = min(2, len(suites) or 1)
    nrows = int(np.ceil((len(suites) or 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, suite in zip(axes.ravel(), suites):
        ax.axis("on")
        suite_rows = [row for row in rows if row.get("suite") == suite]
        policies = _ordered_replacement_policies({row["policy"] for row in suite_rows if row.get("policy")})
        for policy in policies:
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in suite_rows:
                if row.get("policy") != policy:
                    continue
                grouped[float(row["global_budget_fraction"])].append(float(row["verified_successes"]))
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            ax.plot(
                [x * 100.0 for x in xs],
                ys,
                marker=_replacement_policy_marker(policy),
                linewidth=2.2,
                markersize=4.8,
                label=policy,
                color=_replacement_policy_color(policy),
            )
        ax.set_title(_suite_label(suite), fontweight="bold")
        ax.set_xlabel("Global budget fraction (%)")
        ax.set_ylabel("Verified successes")
        ax.set_xticks([25, 50, 75])
        ax.grid(True, alpha=0.35)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Strict fixed-budget scheduling on replacement tracks", fontsize=13.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return _save(fig, prefix)


def _ordered_replacement_policies(policies: set[str]) -> list[str]:
    order = {
        "oracle": 0,
        "raw self-forecast": 1,
        "recalibrated self-forecast": 2,
        "source-by-budget prior": 3,
        "constant-by-budget prior": 4,
        "random task order": 5,
    }
    return sorted(policies, key=lambda policy: (order.get(policy, 99), policy))


def _replacement_policy_color(policy: str) -> str:
    return {
        "oracle": "#111827",
        "raw self-forecast": "#2563eb",
        "recalibrated self-forecast": "#059669",
        "source-by-budget prior": "#7c3aed",
        "constant-by-budget prior": "#d97706",
        "random task order": "#6b7280",
    }.get(policy, "#4b5563")


def _replacement_policy_marker(policy: str) -> str:
    return {
        "oracle": "o",
        "raw self-forecast": "D",
        "recalibrated self-forecast": "s",
        "source-by-budget prior": "^",
        "constant-by-budget prior": "v",
        "random task order": "X",
    }.get(policy, "o")


def plot_fixed_budget_scheduling(
    *,
    table_path: str | Path = "reports/tables/paper_table15_fixed_budget_scheduling.csv",
    figures_dir: str | Path = "reports/figures",
) -> list[Path]:
    rows = _read_csv(Path(table_path))
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suites = sorted({row["suite"] for row in rows if row.get("suite")})
    if not rows or not suites:
        return _save_empty(figures_dir / "paper_figure11_fixed_budget_scheduling", "No strict fixed-budget scheduling rows")
    ncols = min(3, len(suites))
    nrows = int(np.ceil(len(suites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, suite in zip(axes.ravel(), suites):
        ax.axis("on")
        suite_rows = [row for row in rows if row.get("suite") == suite]
        for method in FIXED_BUDGET_METHODS:
            method_rows = [row for row in suite_rows if row.get("method") == method]
            if not method_rows:
                continue
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in method_rows:
                grouped[float(row["budget_fraction"])].append(float(row["verified_successes"]))
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            ax.plot(
                [x * 100.0 for x in xs],
                ys,
                marker=_marker(method),
                linewidth=2.0,
                markersize=4.4,
                label=_method_label(method),
                color=_color(method),
            )
        ax.set_title(_suite_label(suite), fontweight="bold")
        ax.set_xlabel("Budget fraction of full policy-B frontier (%)")
        ax.set_ylabel("Verified successes")
        ax.set_xticks([25, 50, 75, 100])
        ax.grid(True, alpha=0.35)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Strict fixed-budget scheduling under the same global token budget", fontsize=13.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    return _save(fig, figures_dir / "paper_figure11_fixed_budget_scheduling")


def _curves_by_method(
    run: Any,
    eval_task_ids: list[str],
    eval_outcomes: dict[str, dict[int, bool]],
    *,
    split_map: dict[str, str],
) -> dict[str, dict[str, dict[int, float]]]:
    eval_task_set = set(eval_task_ids)
    calibration_task_ids = {task_id for task_id, split in split_map.items() if split == "calibration"}
    calibration_outcomes = [row for row in run.outcomes if str(row.get("task_id")) in calibration_task_ids]
    calibration_forecasts = [row for row in run.forecasts if str(row.get("task_id")) in calibration_task_ids]
    eval_forecasts = [row for row in run.forecasts if str(row.get("task_id")) in eval_task_set]
    self_curves = forecast_curves(eval_forecasts)
    budget_grid_by_task = {task_id: sorted(eval_outcomes[task_id]) for task_id in eval_task_ids if task_id in eval_outcomes}
    all_budgets = sorted({budget for grid in budget_grid_by_task.values() for budget in grid})
    metadata = _task_records_by_id(run)

    curves: dict[str, dict[str, dict[int, float]]] = {
        "self_forecast_raw": self_curves,
        "single_budget_midpoint": {
            task_id: {
                int(budget): probability
                for budget, probability in single_budget_curve(
                    budget_grid_by_task[task_id][len(budget_grid_by_task[task_id]) // 2],
                    budget_grid_by_task[task_id],
                ).items()
            }
            for task_id in eval_task_ids
            if task_id in budget_grid_by_task
        },
    }
    calibrator = fit_histogram_recalibrator(calibration_forecasts, calibration_outcomes)
    curves["self_forecast_histogram_recalibrated"] = apply_histogram_recalibrator(eval_forecasts, calibrator)
    constant_fit = fit_constant_by_budget(calibration_outcomes)
    curves["constant_by_budget_calibration"] = predict_constant_by_budget(eval_task_ids, all_budgets, constant_fit)
    source_fit = fit_source_by_budget(calibration_outcomes, metadata)
    curves["source_by_budget_calibration"] = predict_source_by_budget(eval_task_ids, metadata, budget_grid_by_task, source_fit)
    prompt_fit = fit_prompt_length_bins(calibration_outcomes, metadata)
    curves["prompt_length_bin_calibration"] = predict_prompt_length_bins(eval_task_ids, metadata, budget_grid_by_task, prompt_fit)
    return {method: _filter_curves_to_outcomes(method_curves, eval_outcomes) for method, method_curves in curves.items()}


def _allocate_policy_a(curves: dict[str, dict[int, float]], capacity: int) -> dict[str, int]:
    candidates: list[tuple[float, float, int, str, float]] = []
    for task_id, curve in curves.items():
        for budget, probability in sorted(curve.items()):
            if budget <= 0:
                continue
            score = float(probability) / float(budget)
            candidates.append((score, float(probability), int(budget), task_id, float(probability)))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    selected: dict[str, tuple[int, float]] = {}
    used = 0
    for _score, _prob, budget, task_id, probability in candidates:
        current = selected.get(task_id)
        if current is None:
            if probability <= 0 or used + budget > capacity:
                continue
            selected[task_id] = (budget, probability)
            used += budget
            continue
        current_budget, current_probability = current
        if budget <= current_budget or probability <= current_probability:
            continue
        added_cost = budget - current_budget
        marginal_score = (probability - current_probability) / added_cost
        if marginal_score > 0 and used + added_cost <= capacity:
            selected[task_id] = (budget, probability)
            used += added_cost
    return {task_id: budget for task_id, (budget, _probability) in selected.items()}


def _allocate_policy_b(curves: dict[str, dict[int, float]], capacity: int) -> dict[str, int]:
    grids = {task_id: sorted(curve) for task_id, curve in curves.items() if curve}
    positions = {task_id: -1 for task_id in grids}
    selected: dict[str, int] = {}
    used = 0

    heap: list[tuple[float, float, int, str, int, int]] = []

    def push_next(task_id: str) -> None:
        budgets = grids[task_id]
        current_index = positions[task_id]
        current_budget = 0 if current_index < 0 else budgets[current_index]
        current_probability = 0.0 if current_budget == 0 else float(curves[task_id][current_budget])
        for next_index in range(current_index + 1, len(budgets)):
            next_budget = budgets[next_index]
            next_probability = float(curves[task_id][next_budget])
            delta_cost = int(next_budget - current_budget)
            delta_p = next_probability - current_probability
            if delta_cost > 0 and delta_p > 1e-12:
                score = delta_p / delta_cost
                heapq.heappush(heap, (-score, -delta_p, delta_cost, task_id, next_budget, next_index))
                return

    for task_id in grids:
        push_next(task_id)

    while heap:
        _neg_score, _neg_delta_p, _delta_cost, task_id, next_budget, next_index = heapq.heappop(heap)
        if next_index <= positions[task_id]:
            continue
        current_index = positions[task_id]
        current_budget = 0 if current_index < 0 else grids[task_id][current_index]
        current_probability = 0.0 if current_budget == 0 else float(curves[task_id][current_budget])
        next_probability = float(curves[task_id][next_budget])
        delta_cost = int(next_budget - current_budget)
        delta_p = next_probability - current_probability
        if delta_cost <= 0 or delta_p <= 1e-12:
            push_next(task_id)
            continue
        if used + delta_cost > capacity:
            continue
        used += delta_cost
        selected[task_id] = int(next_budget)
        positions[task_id] = next_index
        push_next(task_id)
    return selected


def _filter_curves_to_outcomes(
    curves: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, dict[int, float]]:
    filtered: dict[str, dict[int, float]] = {}
    for task_id, task_outcomes in outcomes.items():
        curve = curves.get(task_id)
        if not curve:
            continue
        allowed = set(task_outcomes)
        row = {int(budget): _clamp_probability(probability) for budget, probability in curve.items() if int(budget) in allowed}
        if row:
            filtered[task_id] = dict(sorted(row.items()))
    return filtered


def _special_candidates(method: str, outcomes: dict[str, dict[int, bool]], *, seed: int) -> list[tuple[str, int]]:
    task_ids = sorted(outcomes)
    if method == "cheapest_budget":
        return [(task_id, min(outcomes[task_id])) for task_id in task_ids]
    if method == "max_budget":
        return [(task_id, max(outcomes[task_id])) for task_id in task_ids]
    if method == "random_budget":
        rng = random.Random(seed)
        candidates = [(task_id, rng.choice(sorted(outcomes[task_id]))) for task_id in task_ids]
        rng.shuffle(candidates)
        return candidates
    raise ValueError(f"Unknown special allocation method: {method}")


def _frontier_capacities(outcomes: dict[str, dict[int, bool]], n_points: int = 11) -> list[int]:
    max_total = sum(max(grid) for grid in outcomes.values() if grid)
    if max_total <= 0:
        return [0]
    raw = [0] + [int(round(max_total * fraction)) for fraction in np.linspace(0.1, 1.0, n_points - 1)]
    return sorted(set(raw))


def _verified_successes(allocation: dict[str, int], outcomes: dict[str, dict[int, bool]]) -> int:
    return sum(1 for task_id, budget in allocation.items() if bool(outcomes.get(task_id, {}).get(int(budget), False)))


def _task_records_by_id(run: Any) -> dict[str, TaskRecord]:
    metadata = task_metadata(list(run.forecasts) + list(run.outcomes))
    task_file = run.config.get("task_file")
    if task_file and Path(task_file).exists():
        for row in read_jsonl(task_file):
            task = TaskRecord.model_validate(row)
            metadata.setdefault(task.task_id, {}).update(
                {
                    "track": task.track,
                    "source": task.source,
                    "source_version": task.source_version,
                    "external_id": task.external_id,
                    "prompt": task.prompt,
                    "prompt_length": len(task.prompt),
                }
            )
    result: dict[str, TaskRecord] = {}
    for task_id, meta in metadata.items():
        track = str(meta.get("track") or "math")
        if track not in {"math", "coding", "coding_edit", "code_editing", "swe", "agentic"}:
            track = "math"
        result[task_id] = TaskRecord(
            task_id=task_id,
            track=track,
            prompt=str(meta.get("prompt") or "unavailable prompt"),
            verifier=str(meta.get("verifier") or "numeric_exact"),
            source=str(meta.get("source") or "unknown"),
            source_version=str(meta.get("source_version") or meta.get("source") or "unknown"),
            external_id=str(meta.get("external_id") or task_id),
        )
    return result


def _load_split_map(split_dir: Path, suite: str) -> dict[str, str]:
    path = split_dir / f"{suite}_calibration_eval_split.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_splits = payload.get("task_splits")
    if not isinstance(task_splits, dict):
        return {}
    return {str(task_id): str(split) for task_id, split in task_splits.items()}


def _plot_policy(rows: list[dict[str, str]], policy: str, output_prefix: Path) -> list[Path]:
    policy_rows = [row for row in rows if row.get("policy") == policy]
    suites = sorted({row["suite"] for row in policy_rows if row.get("suite")})
    if not policy_rows or not suites:
        return _save_empty(output_prefix, "No allocation frontier rows")
    ncols = min(3, len(suites))
    nrows = int(np.ceil(len(suites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.55 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, suite in zip(axes.ravel(), suites):
        ax.axis("on")
        suite_rows = [row for row in policy_rows if row.get("suite") == suite]
        methods = _figure_methods(suite_rows)
        for method in methods:
            method_rows = [row for row in suite_rows if row.get("method") == method]
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in method_rows:
                grouped[int(float(row["total_budget"]))].append(float(row["verified_successes"]))
            if not grouped:
                continue
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            ax.plot(xs, ys, marker=_marker(method), linewidth=2.0, markersize=4.2, label=_method_label(method), color=_color(method))
        ax.set_title(_suite_label(suite), fontweight="bold")
        ax.set_xlabel("Total allocated generated-token budget")
        ax.set_ylabel("Verified successes")
        ax.grid(True, alpha=0.35)
        ax.ticklabel_format(style="plain", axis="x")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, -0.02))
    title = "Allocation frontier under expected marginal gain" if policy == "policy_b" else "Allocation frontier under expected utility per budget"
    fig.suptitle(title, fontsize=13.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return _save(fig, output_prefix)


def _figure_methods(rows: list[dict[str, str]]) -> list[str]:
    methods = ["self_forecast_raw", "self_forecast_histogram_recalibrated"]
    simple = _best_simple_prior(rows)
    if simple:
        methods.append(simple)
    methods.extend(["oracle", "random_budget", "max_budget"])
    return [method for method in methods if any(row.get("method") == method for row in rows)]


def _best_simple_prior(rows: list[dict[str, str]]) -> str | None:
    best: tuple[float, float, str] | None = None
    for method in SIMPLE_PRIORS:
        method_rows = [row for row in rows if row.get("method") == method]
        if not method_rows:
            continue
        max_budget = max(int(float(row["total_budget"])) for row in method_rows)
        final_rows = [row for row in method_rows if int(float(row["total_budget"])) == max_budget]
        successes = sum(float(row["verified_successes"]) for row in final_rows) / len(final_rows)
        regret = sum(float(row["regret_to_oracle"]) for row in final_rows) / len(final_rows)
        candidate = (successes, -regret, method)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def _best_simple_prior_by_suite_model(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    selected = _best_simple_prior_from_baseline_table()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        suite = str(row.get("suite") or "")
        model = str(row.get("model") or "")
        if (suite, model) not in selected:
            grouped[(suite, model)].append(row)
    for key, group_rows in grouped.items():
        fallback = _best_simple_prior(group_rows)
        if fallback in FIXED_BUDGET_SIMPLE_PRIORS:
            selected[key] = fallback
        else:
            selected[key] = "constant_by_budget_calibration"
    return selected


def _best_simple_prior_from_baseline_table(
    path: str | Path = "reports/tables/paper_table4_main_baseline_summary.csv",
) -> dict[tuple[str, str], str]:
    table = Path(path)
    if not table.exists() or not table.read_text(encoding="utf-8").strip():
        return {}
    result: dict[tuple[str, str], str] = {}
    for row in _read_csv(table):
        method = str(row.get("best_simple_prior_method") or "")
        if method in FIXED_BUDGET_SIMPLE_PRIORS:
            result[(str(row.get("suite") or ""), str(row.get("model") or ""))] = method
    return result


def _method_label(method: str) -> str:
    return {
        "self_forecast_raw": "Raw self-forecast",
        "self_forecast_histogram_recalibrated": "Recalibrated self-forecast",
        "constant_by_budget_calibration": "Best simple prior: budget",
        "source_by_budget_calibration": "Best simple prior: source",
        "prompt_length_bin_calibration": "Best simple prior: prompt length",
        "single_budget_midpoint": "Best simple prior: midpoint",
        "best_simple_prior": "Best simple prior",
        "oracle": "Oracle upper bound",
        "random_budget": "Random budget",
        "max_budget": "Max budget",
    }.get(method, method.replace("_", " "))


def _replacement_policy_label(method: str) -> str:
    return {
        "random_budget": "random task order",
        "constant_by_budget_calibration": "constant-by-budget prior",
        "source_by_budget_calibration": "source-by-budget prior",
        "self_forecast_raw": "raw self-forecast",
        "self_forecast_histogram_recalibrated": "recalibrated self-forecast",
        "oracle": "oracle",
    }.get(method, method.replace("_", " "))


def _color(method: str) -> str:
    return {
        "self_forecast_raw": "#2563eb",
        "self_forecast_histogram_recalibrated": "#059669",
        "constant_by_budget_calibration": "#92400e",
        "source_by_budget_calibration": "#b45309",
        "prompt_length_bin_calibration": "#d97706",
        "single_budget_midpoint": "#a16207",
        "best_simple_prior": "#d97706",
        "oracle": "#111827",
        "random_budget": "#7c3aed",
        "max_budget": "#dc2626",
    }.get(method, "#4b5563")


def _marker(method: str) -> str:
    return {
        "self_forecast_raw": "o",
        "self_forecast_histogram_recalibrated": "s",
        "best_simple_prior": "P",
        "oracle": "D",
        "random_budget": "^",
        "max_budget": "v",
    }.get(method, "P")


def _suite_label(suite: str) -> str:
    return {
        "paper_math_core": "Math",
        "paper_evalplus_humaneval_full": "HumanEval+",
        "paper_evalplus_mbpp_full": "MBPP+",
        "paper_bigcodebench_hard": "BigCodeBench-Hard",
        "paper_aider_polyglot": "Aider Polyglot",
    }.get(suite, suite.replace("paper_", "").replace("_", " "))


def _save(fig: Any, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".png"), prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return paths


def _save_empty(prefix: Path, message: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, prefix)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv_with_fields(path, rows, TABLE_FIELDS)


def _write_csv_with_fields(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _artifact_roots(value: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]
    return [Path(value)]


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    return _as_int(value)


def _select_at_or_below_target(rows: list[dict[str, Any]], target_budget: int) -> dict[str, Any]:
    """Return the highest-budget row that does not exceed target_budget."""

    if not rows:
        raise ValueError("Cannot select fixed-budget point from empty rows")

    sorted_rows = sorted(rows, key=lambda row: _as_int(row.get("total_budget")))
    candidates = [row for row in sorted_rows if _as_int(row.get("total_budget")) <= target_budget]
    if candidates:
        return max(candidates, key=lambda row: _as_int(row.get("total_budget")))

    zero_rows = [row for row in sorted_rows if _as_int(row.get("total_budget")) == 0]
    if zero_rows:
        return zero_rows[0]

    template = sorted_rows[0].copy()
    template["total_budget"] = 0
    template["allocated_tasks"] = 0
    template["verified_successes"] = 0
    template["success_rate"] = 0.0
    template["budget_used"] = 0
    template["oracle_successes"] = 0
    template["regret_to_oracle"] = 0
    return template


def _strict_budget_feasible(
    *,
    target_total_budget: int,
    selected_total_budget: int,
    budget_used: int,
    budget_slack_tokens: int,
) -> bool:
    return (
        selected_total_budget <= target_total_budget
        and budget_used <= selected_total_budget
        and budget_used <= target_total_budget
        and budget_slack_tokens >= 0
    )


def _clamp_probability(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate global-budget allocation frontiers from frozen forecasts.")
    parser.add_argument("--artifact-root", action="append", default=None)
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--output-table", default="reports/tables/paper_table12_allocation_frontier.csv")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--write-fixed-budget-table", default=None)
    parser.add_argument("--suite", action="append", default=None, help="Restrict analysis to a suite; repeatable.")
    args = parser.parse_args()
    outputs = run_allocation_frontier(
        artifact_root=args.artifact_root or ["reports/artifacts"],
        split_dir=args.split_dir,
        output_table=args.output_table,
        figures_dir=args.figures_dir,
        seed=args.seed,
        write_figures=not args.no_figures,
        fixed_budget_table=args.write_fixed_budget_table,
        suite_filter=set(args.suite) if args.suite else None,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
