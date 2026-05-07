#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.metrics.regret import oracle_utility, selected_budget_from_forecast, utility
from budget2success.metrics.first_success_budget import (
    censored_lower_bound_error,
    log_token_error,
    max_budget_failure_rate,
    observed_censored_at_budget,
    observed_first_success_budget,
    observed_budget2success,
    overbudget_ratio,
    pairwise_ranking_accuracy,
    underbudgeted,
)
from budget2success.schemas.records import ExperimentConfig
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl
from budget2success.utils.manifest import write_redacted_config_snapshot, write_run_manifest
try:
    from run_experiment_suite import suite_model_configs
except ImportError:  # pragma: no cover - package import path used by tests.
    from scripts.run_experiment_suite import suite_model_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TokenCapBench forecasts and outcomes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-root", default=None, help="Score every completed run under this suite/run directory.")
    parser.add_argument("--token-cost", type=float, default=0.0)
    parser.add_argument("--models", nargs="+", default=None, help="Override configured models for suite configs.")
    args = parser.parse_args()
    if args.run_root:
        _score_run_root(Path(args.run_root), token_cost=args.token_cost)
        return
    if not args.config:
        parser.error("Specify --config or --run-root.")
    raw_config = load_yaml(args.config)
    if _is_suite_config(raw_config):
        _run_suite_scoring(args.config, raw_config, args)
        return
    cfg = ExperimentConfig.model_validate(raw_config)
    run_dir = Path(cfg.output_dir) / cfg.run_id
    write_redacted_config_snapshot(raw_config, run_dir / "config_snapshot.yaml")
    forecasts = read_jsonl(run_dir / "forecasts.jsonl")
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")

    forecast_by_task = {row["task_id"]: row for row in forecasts if "p_success_by_budget" in row}
    outcomes_by_task: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in outcomes:
        outcomes_by_task[row["task_id"]][int(row["budget"])] = bool(row["success"])

    probs: list[float] = []
    ys: list[bool] = []
    token_errors: list[float] = []
    lower_bound_errors: list[float] = []
    underbudget_flags: list[bool] = []
    overbudget_ratios: list[float] = []
    regrets: list[float] = []
    observed_ttg_by_task: dict[str, int | None] = {}
    predicted_ttg_by_task: dict[str, float | None] = {}

    for task_id, forecast in forecast_by_task.items():
        task_outcomes = outcomes_by_task.get(task_id, {})
        if not task_outcomes:
            continue
        p_by_budget = {int(k): float(v) for k, v in forecast["p_success_by_budget"].items()}
        for budget, prob in p_by_budget.items():
            if budget in task_outcomes:
                probs.append(prob)
                ys.append(task_outcomes[budget])
        obs = observed_first_success_budget(task_outcomes)
        censored_at = observed_censored_at_budget(task_outcomes)
        pred = forecast.get("median_budget2success")
        observed_ttg_by_task[task_id] = obs
        predicted_ttg_by_task[task_id] = pred
        err = log_token_error(pred, obs)
        if err is not None:
            token_errors.append(err)
        lb_err = censored_lower_bound_error(pred, censored_at)
        if lb_err is not None:
            lower_bound_errors.append(lb_err)
        under = underbudgeted(pred, obs)
        if under is not None:
            underbudget_flags.append(under)
        over = overbudget_ratio(pred, obs)
        if over is not None:
            overbudget_ratios.append(over)
        if p_by_budget:
            chosen = selected_budget_from_forecast(p_by_budget, reward=1.0, token_cost=args.token_cost)
            oracle = oracle_utility(task_outcomes, reward=1.0, token_cost=args.token_cost)
            chosen_success = task_outcomes.get(chosen, False)
            regrets.append(oracle - utility(chosen_success, chosen, reward=1.0, token_cost=args.token_cost))

    all_observed_ttg = {task_id: observed_budget2success(task_outcomes) for task_id, task_outcomes in outcomes_by_task.items()}
    censored_count = sum(1 for value in all_observed_ttg.values() if value is None)
    max_fail_rate = max_budget_failure_rate(outcomes_by_task)
    wall_times = [float(row["wall_time_seconds"]) for row in outcomes if row.get("wall_time_seconds") is not None]
    completion_tokens = [int(row["completion_tokens"]) for row in outcomes if row.get("completion_tokens") is not None]

    metrics = {
        "n_forecasts": len(forecast_by_task),
        "n_outcome_tasks": len(outcomes_by_task),
        "n_budget_observations": len(ys),
        "brier": _finite_or_none(brier_score(probs, ys)),
        "ece": _finite_or_none(expected_calibration_error(probs, ys)),
        "mean_log_token_error": sum(token_errors) / len(token_errors) if token_errors else None,
        "solved_only_log_ttg_error": sum(token_errors) / len(token_errors) if token_errors else None,
        "censored_lower_bound_error": sum(lower_bound_errors) / len(lower_bound_errors) if lower_bound_errors else None,
        "underbudget_rate": sum(underbudget_flags) / len(underbudget_flags) if underbudget_flags else None,
        "mean_overbudget_ratio": sum(overbudget_ratios) / len(overbudget_ratios) if overbudget_ratios else None,
        "overbudget_ratio": sum(overbudget_ratios) / len(overbudget_ratios) if overbudget_ratios else None,
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        "regret": sum(regrets) / len(regrets) if regrets else None,
        "censoring_rate": censored_count / len(all_observed_ttg) if all_observed_ttg else None,
        "max_budget_failure_rate": max_fail_rate,
        "pairwise_ttg_ranking_accuracy": pairwise_ranking_accuracy(predicted_ttg_by_task, observed_ttg_by_task),
        "mean_wall_time_seconds": sum(wall_times) / len(wall_times) if wall_times else None,
        "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
    }
    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_manifest(
        run_dir,
        config=raw_config,
        command_line_arguments=sys.argv[1:],
        phase="scored",
        extra={"metrics_file": str(out_path)},
    )
    print(json.dumps(metrics, indent=2))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _is_suite_config(config: dict) -> bool:
    return "run_id" not in config and bool(config.get("suite_name") or config.get("suite") or config.get("models"))


def _run_suite_scoring(config_path: str, suite_cfg: dict, args: argparse.Namespace) -> None:
    suite_name = str(suite_cfg.get("suite_name") or suite_cfg.get("suite") or Path(config_path).stem)
    output_root = Path(str(suite_cfg.get("output_root") or "reports/runs"))
    for model_cfg in suite_model_configs(suite_cfg, suite_name=suite_name, models_override=args.models):
        run_dir = output_root / suite_name / model_cfg["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = run_dir / "config_snapshot.yaml"
        write_redacted_config_snapshot(model_cfg, snapshot)
        command = [
            sys.executable,
            "scripts/score_results.py",
            "--config",
            str(snapshot),
            "--token-cost",
            str(args.token_cost),
        ]
        subprocess.run(command, check=True)


def _score_run_root(run_root: Path, *, token_cost: float) -> None:
    if (run_root / "forecasts.jsonl").exists() and (run_root / "outcomes.jsonl").exists():
        candidates = [run_root]
    else:
        candidates = [
            path
            for path in sorted(run_root.iterdir()) if path.is_dir() and (path / "forecasts.jsonl").exists() and (path / "outcomes.jsonl").exists()
        ]
    for run_dir in candidates:
        config_path = run_dir / "config_snapshot.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot score {run_dir}: missing config_snapshot.yaml")
        subprocess.run(
            [
                sys.executable,
                "scripts/score_results.py",
                "--config",
                str(config_path),
                "--token-cost",
                str(token_cost),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
