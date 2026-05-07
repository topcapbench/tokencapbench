#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import (
    forecast_curves,
    forecast_medians,
    load_paper_runs,
    metric_ci,
    outcomes_by_task,
    score_curve_set,
    task_metadata,
)


METRICS = [
    "brier",
    "ece",
    "success_at_max_budget",
    "censoring_rate",
    "solved_only_log_ttg_error",
    "signed_log_budget_error_mean",
    "absolute_log_budget_error_mean",
    "censored_lower_bound_error",
    "max_budget_failure_rate",
    "underbudget_rate",
    "overbudget_rate",
    "underbudget_shortfall_factor_mean",
    "overbudget_waste_factor_mean",
    "regret",
    "normalized_regret",
    "forecast_monotonicity_violation_rate",
    "outcome_nonmonotonicity_rate",
    "task_budget_ranking_accuracy",
    "truncation_rate",
]


def bootstrap_suite(
    *,
    suite: str | None = None,
    run_dir: str | Path | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    output_dir: str | Path = "reports/tables",
    artifact_root: str | Path | None = "reports/artifacts",
    include_artifacts: bool = True,
    corrected_artifact_root: str | Path | None = None,
    math_label_mode: str = "original",
    ranking_max_pairs: int = 10000,
    ranking_seed: int = 20260428,
) -> tuple[Path, Path]:
    runs = load_paper_runs(
        suite=suite,
        run_dirs=[run_dir] if run_dir else None,
        run_root=Path(artifact_root) / "__no_reports_runs__" if artifact_root is not None and run_dir is None else "reports/runs",
        artifact_root=artifact_root,
        include_artifacts=include_artifacts,
        corrected_artifact_root=corrected_artifact_root,
        math_label_mode=math_label_mode,
    )
    if not runs:
        raise FileNotFoundError(f"No run artifacts found for suite={suite!r} run_dir={run_dir!r}")
    rng = np.random.default_rng(seed)
    main_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    for run in runs:
        run_suite = suite if suite is not None else run.suite or ""
        curves = forecast_curves(run.forecasts)
        medians = forecast_medians(run.forecasts)
        outcomes = outcomes_by_task(run.outcomes)
        outcome_rows_by_task = _outcome_rows_by_task(run.outcomes)
        metadata = task_metadata(list(run.forecasts) + list(run.outcomes))
        groups = _groups(metadata, set(curves) & set(outcomes))
        metric_groups = {("all", "all"): groups.get(("all", "all"), sorted(set(curves) & set(outcomes)))}
        for (track, source), task_ids in metric_groups.items():
            main_rows.extend(
                _bootstrap_metric_group(
                    suite=run_suite,
                    run_id=run.run_id,
                    model=run.model,
                    track=track,
                    source=source,
                    task_ids=task_ids,
                    curves=curves,
                    medians=medians,
                    outcomes=outcomes,
                    outcome_rows_by_task=outcome_rows_by_task,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    rng=rng,
                    ranking_max_pairs=ranking_max_pairs,
                    ranking_seed=ranking_seed,
                )
            )
        for (track, source), task_ids in groups.items():
            success_rows.extend(
                _bootstrap_success_group(
                    suite=run_suite,
                    run_id=run.run_id,
                    model=run.model,
                    track=track,
                    source=source,
                    task_ids=task_ids,
                    outcomes=outcomes,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    rng=rng,
                )
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "bootstrap_main_metrics.csv"
    success_path = output_dir / "bootstrap_success_by_budget.csv"
    _write_csv(main_path, main_rows)
    _write_csv(success_path, success_rows)
    return main_path, success_path


def _groups(metadata: dict[str, dict[str, Any]], task_ids: set[str]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {("all", "all"): sorted(task_ids)}
    for task_id in sorted(task_ids):
        meta = metadata.get(task_id, {})
        track = str(meta.get("track") or "unknown")
        source = str(meta.get("source") or "unknown")
        groups.setdefault((track, source), []).append(task_id)
    return groups


def _bootstrap_metric_group(
    *,
    suite: str,
    run_id: str,
    model: str,
    track: str,
    source: str,
    task_ids: list[str],
    curves: dict[str, dict[int, float]],
    medians: dict[str, float | None],
    outcomes: dict[str, dict[int, bool]],
    outcome_rows_by_task: dict[str, list[dict[str, Any]]],
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
    ranking_max_pairs: int,
    ranking_seed: int,
) -> list[dict[str, Any]]:
    if not task_ids:
        return []
    base = _sample_score(
        task_ids,
        curves,
        medians,
        outcomes,
        outcome_rows_by_task,
        ranking_max_pairs=ranking_max_pairs,
        ranking_seed=ranking_seed,
    )
    metric_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sample = rng.choice(task_ids, size=len(task_ids), replace=True).tolist()
        scored = _sample_score(
            sample,
            curves,
            medians,
            outcomes,
            outcome_rows_by_task,
            include_pairwise=True,
            ranking_max_pairs=ranking_max_pairs,
            ranking_seed=ranking_seed + _,
        )
        for metric in METRICS:
            value = scored.get(metric)
            if value is not None and np.isfinite(float(value)):
                metric_values[metric].append(float(value))
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        values = metric_values.get(metric, [])
        mean, low, high = metric_ci(values, confidence) if values else (base.get(metric), None, None)
        rows.append(
            {
                "suite": suite,
                "run_id": run_id,
                "model": model,
                "track": track,
                "source": source,
                "metric": metric,
                "estimate": base.get(metric),
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "confidence": confidence,
                "n_tasks": len(task_ids),
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _sample_score(
    sample_task_ids: list[str],
    curves: dict[str, dict[int, float]],
    medians: dict[str, float | None],
    outcomes: dict[str, dict[int, bool]],
    outcome_rows_by_task: dict[str, list[dict[str, Any]]] | None = None,
    include_pairwise: bool = True,
    ranking_max_pairs: int = 10000,
    ranking_seed: int = 20260428,
) -> dict[str, Any]:
    sample_curves: dict[str, dict[int, float]] = {}
    sample_medians: dict[str, float | None] = {}
    sample_outcomes: dict[str, dict[int, bool]] = {}
    sample_rows: list[dict[str, Any]] = []
    for index, task_id in enumerate(sample_task_ids):
        key = f"{task_id}__sample_{index}"
        sample_curves[key] = curves[task_id]
        sample_medians[key] = medians.get(task_id)
        sample_outcomes[key] = outcomes[task_id]
        if outcome_rows_by_task:
            sample_rows.extend(outcome_rows_by_task.get(task_id, []))
    return score_curve_set(
        sample_curves,
        sample_outcomes,
        predicted_ttg_by_task=sample_medians,
        outcome_rows=sample_rows,
        include_pairwise=include_pairwise,
        ranking_max_pairs=ranking_max_pairs,
        ranking_seed=ranking_seed,
    )


def _outcome_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return dict(grouped)


def _bootstrap_success_group(
    *,
    suite: str,
    run_id: str,
    model: str,
    track: str,
    source: str,
    task_ids: list[str],
    outcomes: dict[str, dict[int, bool]],
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    budgets = sorted({budget for task_id in task_ids for budget in outcomes.get(task_id, {})})
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        observed_values = [outcomes[task_id][budget] for task_id in task_ids if budget in outcomes.get(task_id, {})]
        if not observed_values:
            continue
        boot_values: list[float] = []
        eligible_task_ids = [task_id for task_id in task_ids if budget in outcomes.get(task_id, {})]
        for _ in range(n_bootstrap):
            sample = rng.choice(eligible_task_ids, size=len(eligible_task_ids), replace=True)
            boot_values.append(float(np.mean([outcomes[str(task_id)][budget] for task_id in sample])))
        mean, low, high = metric_ci(boot_values, confidence)
        rows.append(
            {
                "suite": suite,
                "run_id": run_id,
                "model": model,
                "track": track,
                "source": source,
                "budget": budget,
                "success_rate": float(np.mean(observed_values)),
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
                "confidence": confidence,
                "n_tasks": len(observed_values),
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap TokenCapBench metrics over tasks.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not discover packaged reports/artifacts runs.")
    parser.add_argument("--corrected-artifact-root", default=None)
    parser.add_argument(
        "--math-label-mode",
        choices=["original", "strict", "corrected"],
        default="original",
        help="'strict' is retained as an alias for task-default corrected math labels.",
    )
    parser.add_argument("--ranking-max-pairs", type=int, default=10000)
    parser.add_argument("--ranking-seed", type=int, default=20260428)
    args = parser.parse_args()
    main_path, success_path = bootstrap_suite(
        suite=args.suite,
        run_dir=args.run_dir,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        output_dir=args.output_dir,
        artifact_root=args.artifact_root,
        include_artifacts=not args.no_artifacts,
        corrected_artifact_root=args.corrected_artifact_root,
        math_label_mode=args.math_label_mode,
        ranking_max_pairs=args.ranking_max_pairs,
        ranking_seed=args.ranking_seed,
    )
    print(f"Wrote {main_path}")
    print(f"Wrote {success_path}")


if __name__ == "__main__":
    main()
