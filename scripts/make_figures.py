#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from collections import defaultdict
from pathlib import Path

from budget2success.analysis.plots import (
    regret_curve_plot,
    reliability_plot,
    scatter_predicted_observed,
    success_rate_by_budget_plot,
)
from budget2success.execution.token_budget import observed_budget2success
from budget2success.metrics.regret import oracle_utility, selected_budget_from_forecast, utility
from budget2success.schemas.records import ExperimentConfig
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Make simple paper figures from a run.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.model_validate(load_yaml(args.config))
    run_dir = Path(cfg.output_dir) / cfg.run_id
    forecasts = read_jsonl(run_dir / "forecasts.jsonl")
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")
    forecast_by_task = {row["task_id"]: row for row in forecasts if "p_success_by_budget" in row}
    outcomes_by_task: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in outcomes:
        outcomes_by_task[row["task_id"]][int(row["budget"])] = bool(row["success"])

    probs: list[float] = []
    ys: list[bool] = []
    pred_ttg: list[float] = []
    obs_ttg: list[float] = []
    for task_id, forecast in forecast_by_task.items():
        task_outcomes = outcomes_by_task.get(task_id, {})
        for b_str, p in forecast["p_success_by_budget"].items():
            b = int(b_str)
            if b in task_outcomes:
                probs.append(float(p))
                ys.append(task_outcomes[b])
        obs = observed_budget2success(task_outcomes)
        pred = forecast.get("median_budget2success")
        if obs and pred:
            pred_ttg.append(float(pred))
            obs_ttg.append(float(obs))

    success_series = _success_series_by_group(outcomes, group_key="track")
    success_rate_by_budget_plot(success_series, "reports/figures/figure2_pilot_success_curves.svg")
    success_rate_by_budget_plot(success_series, "reports/figures/figure2_pilot_success_curves.png")
    success_rate_by_budget_plot(success_series, "reports/figures/pilot_success_curves.svg")

    reliability_plot(probs, ys, "reports/figures/figure3_reliability_diagram.svg")
    reliability_plot(probs, ys, "reports/figures/figure3_reliability_diagram.png")
    reliability_plot(probs, ys, "reports/figures/calibration_curves.svg")
    if pred_ttg and obs_ttg:
        scatter_predicted_observed(pred_ttg, obs_ttg, "reports/figures/figure4_tokencapbench_scatter.svg")
        scatter_predicted_observed(pred_ttg, obs_ttg, "reports/figures/figure4_tokencapbench_scatter.png")
        scatter_predicted_observed(pred_ttg, obs_ttg, "reports/figures/tokencapbench_scatter.svg")
    regret_curves = _regret_curves(forecast_by_task, outcomes_by_task)
    regret_curve_plot(regret_curves, "reports/figures/figure5_token_regret_curves.svg")
    regret_curve_plot(regret_curves, "reports/figures/figure5_token_regret_curves.png")

    coding_series = _success_series_by_group(
        [row for row in outcomes if (row.get("metadata") or {}).get("track") == "coding"],
        group_key="source",
    )
    if coding_series:
        success_rate_by_budget_plot(
            coding_series, "reports/figures/figure6_coding_source_robustness.svg", title="Coding Source Robustness"
        )
        success_rate_by_budget_plot(
            coding_series, "reports/figures/figure6_coding_source_robustness.png", title="Coding Source Robustness"
        )

    swe_series = _success_series_by_group(
        [row for row in outcomes if (row.get("metadata") or {}).get("track") == "swe"],
        group_key="source",
    )
    if swe_series:
        success_rate_by_budget_plot(swe_series, "reports/figures/figure7_swe_summary.svg", title="SWE Summary")
        success_rate_by_budget_plot(swe_series, "reports/figures/figure7_swe_summary.png", title="SWE Summary")
    print("Wrote reports/figures")


def _success_series_by_group(rows: list[dict], group_key: str) -> dict[str, dict[int, float]]:
    counts: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        metadata = row.get("metadata") or {}
        group = str(metadata.get(group_key) or metadata.get("track") or "all")
        counts[group][int(row["budget"])].append(bool(row["success"]))
    return {
        group: {budget: sum(values) / len(values) for budget, values in sorted(budget_values.items())}
        for group, budget_values in sorted(counts.items())
    }


def _regret_curves(forecast_by_task: dict[str, dict], outcomes_by_task: dict[str, dict[int, bool]]) -> dict[str, list[tuple[float, float]]]:
    token_costs = [0.0, 1e-5, 1e-4, 1e-3]
    points: list[tuple[float, float]] = []
    for token_cost in token_costs:
        regrets: list[float] = []
        for task_id, forecast in forecast_by_task.items():
            task_outcomes = outcomes_by_task.get(task_id, {})
            if not task_outcomes:
                continue
            p_by_budget = {int(k): float(v) for k, v in forecast["p_success_by_budget"].items()}
            chosen = selected_budget_from_forecast(p_by_budget, reward=1.0, token_cost=token_cost)
            chosen_success = task_outcomes.get(chosen, False)
            oracle = oracle_utility(task_outcomes, reward=1.0, token_cost=token_cost)
            regrets.append(oracle - utility(chosen_success, chosen, reward=1.0, token_cost=token_cost))
        if regrets:
            points.append((token_cost, sum(regrets) / len(regrets)))
    return {"self forecast": points}


if __name__ == "__main__":
    main()
