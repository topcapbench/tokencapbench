#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from collections import defaultdict
from pathlib import Path

from budget2success.analysis.tables import write_csv
from budget2success.baselines.constant_prior import constant_curve
from budget2success.baselines.domain_prior import domain_curve
from budget2success.baselines.prompt_length import prompt_length_curve
from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.schemas.records import ExperimentConfig
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Make simple result tables from a run.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.model_validate(load_yaml(args.config))
    run_dir = Path(cfg.output_dir) / cfg.run_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [{"run_id": cfg.run_id, "model": cfg.model, **metrics}]
    write_csv("reports/tables/main_metrics.csv", rows)
    write_csv("reports/tables/table5_main_metric_summary.csv", rows)

    tasks = load_tasks_jsonl(cfg.task_file)
    if cfg.limit:
        tasks = tasks[: cfg.limit]
    forecasts = read_jsonl(run_dir / "forecasts.jsonl")
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")

    write_csv("reports/tables/table1_substrate_matrix.csv", _table1_substrate_rows())
    write_csv("reports/tables/table2_dataset_composition.csv", _table2_dataset_rows(tasks))
    write_csv("reports/tables/table3_token_time_accounting.csv", _table3_accounting_rows())
    write_csv("reports/tables/table6_baseline_comparison.csv", _table6_baseline_rows(tasks, forecasts, outcomes))
    print("Wrote reports/tables")


def _table1_substrate_rows() -> list[dict[str, str]]:
    return [
        {
            "substrate": "GSM8K",
            "codebase": "openai/grade-school-math or HF openai/gsm8k",
            "task_type": "math word problems",
            "verifier": "numeric exact answer",
            "integration_difficulty": "low",
            "budget2success_role": "pilot and calibration sanity track",
            "decision": "primary",
        },
        {
            "substrate": "MATH / MATH-500",
            "codebase": "hendrycks/math or HF hendrycks/competition_math",
            "task_type": "competition math",
            "verifier": "boxed answer / optional math-verify",
            "integration_difficulty": "medium",
            "budget2success_role": "main math reasoning track",
            "decision": "primary",
        },
        {
            "substrate": "EvalPlus HumanEval+ / MBPP+",
            "codebase": "evalplus/evalplus",
            "task_type": "function-level coding",
            "verifier": "official EvalPlus tests",
            "integration_difficulty": "low",
            "budget2success_role": "coding sanity and baseline track",
            "decision": "primary",
        },
        {
            "substrate": "BigCodeBench",
            "codebase": "bigcode-project/bigcodebench",
            "task_type": "library-heavy Python coding",
            "verifier": "official unit-test evaluator",
            "integration_difficulty": "medium",
            "budget2success_role": "main practical coding track",
            "decision": "secondary",
        },
        {
            "substrate": "LiveCodeBench",
            "codebase": "livecodebench/livecodebench",
            "task_type": "fresh contest coding",
            "verifier": "official contest-style tests",
            "integration_difficulty": "medium",
            "budget2success_role": "freshness and contamination check",
            "decision": "secondary",
        },
        {
            "substrate": "SWE-bench Verified",
            "codebase": "swe-bench/SWE-bench",
            "task_type": "repository-level issue resolution",
            "verifier": "official SWE-bench Docker harness",
            "integration_difficulty": "high",
            "budget2success_role": "flagship SWE track",
            "decision": "flagship",
        },
        {
            "substrate": "BFCL / tau2-bench",
            "codebase": "Gorilla BFCL; sierra-research/tau2-bench",
            "task_type": "tool use and multi-turn agentic tasks",
            "verifier": "official evaluator / simulator success",
            "integration_difficulty": "medium-high",
            "budget2success_role": "optional agentic token/time extension",
            "decision": "optional",
        },
    ]


def _table2_dataset_rows(tasks) -> list[dict[str, str | int]]:
    grouped: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for task in tasks:
        grid = ",".join(str(b) for b in (task.budget_grid or []))
        key = (task.track, task.source, task.source_version or "", task.verifier, grid)
        grouped[key] += 1
    return [
        {
            "track": track,
            "source": source,
            "source_version": source_version,
            "verifier": verifier,
            "budget_grid": budget_grid,
            "n_tasks": n_tasks,
        }
        for (track, source, source_version, verifier, budget_grid), n_tasks in sorted(grouped.items())
    ]


def _table3_accounting_rows() -> list[dict[str, str]]:
    return [
        {
            "field": "budget",
            "definition": "hard generated-token cap passed to the model API or external harness",
            "primary_for": "math and standalone coding budget grids",
        },
        {
            "field": "completion_tokens",
            "definition": "provider-reported visible generated tokens when available",
            "primary_for": "paper budget2success accounting",
        },
        {
            "field": "prompt_tokens",
            "definition": "provider-reported input tokens for the forecast or solver call",
            "primary_for": "secondary accounting and cost analysis",
        },
        {
            "field": "total_tokens",
            "definition": "provider-reported total billable tokens when available",
            "primary_for": "secondary provider cost accounting",
        },
        {
            "field": "reasoning_tokens",
            "definition": "provider-reported hidden reasoning tokens when exposed",
            "primary_for": "reported separately; not mixed with visible generated tokens",
        },
        {
            "field": "wall_time_seconds",
            "definition": "runner-measured elapsed wall-clock time around the generation call",
            "primary_for": "agentic and SWE timing extensions",
        },
    ]


def _table6_baseline_rows(tasks, forecasts: list[dict], outcomes: list[dict]) -> list[dict[str, str | float | int | None]]:
    task_by_id = {task.task_id: task for task in tasks}
    forecast_by_task = {row["task_id"]: row for row in forecasts if "p_success_by_budget" in row}
    outcomes_by_task: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in outcomes:
        outcomes_by_task[row["task_id"]][int(row["budget"])] = bool(row["success"])

    curves_by_baseline: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for task_id, task in task_by_id.items():
        task_outcomes = outcomes_by_task.get(task_id)
        if not task_outcomes:
            continue
        grid = sorted(task_outcomes)
        if task_id in forecast_by_task:
            curves_by_baseline["self_forecast"][task_id] = {
                str(k): float(v) for k, v in forecast_by_task[task_id]["p_success_by_budget"].items()
            }
        curves_by_baseline["constant_0.5"][task_id] = constant_curve(grid, probability=0.5)
        curves_by_baseline["domain_prior"][task_id] = domain_curve(task.track, grid)
        curves_by_baseline["prompt_length"][task_id] = prompt_length_curve(task.prompt, grid)

    rows: list[dict[str, str | float | int | None]] = []
    for baseline, curves in sorted(curves_by_baseline.items()):
        probabilities: list[float] = []
        labels: list[bool] = []
        for task_id, curve in curves.items():
            task_outcomes = outcomes_by_task.get(task_id, {})
            for budget_str, probability in curve.items():
                budget = int(budget_str)
                if budget in task_outcomes:
                    probabilities.append(float(probability))
                    labels.append(bool(task_outcomes[budget]))
        rows.append(
            {
                "baseline": baseline,
                "n_budget_observations": len(labels),
                "brier": brier_score(probabilities, labels) if labels else None,
                "ece": expected_calibration_error(probabilities, labels) if labels else None,
            }
        )
    return rows


if __name__ == "__main__":
    main()
