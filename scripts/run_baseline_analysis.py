#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import (
    forecast_curves,
    load_paper_runs,
    outcomes_by_task,
    score_curve_set,
    ci_string,
    metric_ci,
    task_metadata,
)
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
from budget2success.baselines.constant_prior import constant_curve
from budget2success.baselines.domain_prior import domain_curve
from budget2success.baselines.output_length import output_length_baseline_curve
from budget2success.baselines.prompt_length import prompt_length_curve
from budget2success.baselines.single_budget import single_budget_curve
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl


def run_baseline_analysis(
    *,
    suite: str | None = None,
    run_dir: str | Path | None = None,
    artifact_root: str | Path | None = "reports/artifacts",
    include_artifacts: bool = True,
    output_table: str | Path = "reports/tables/baseline_comparison.csv",
    output_summary: str | Path = "reports/tables/baseline_summary.csv",
    output_figure_prefix: str | Path = "reports/figures/figure_baseline_comparison",
    split_dir: str | Path = "reports/splits",
    use_calibration_split: bool = False,
    include_test_distribution_diagnostics: bool = True,
    calibration_table: str | Path = "reports/tables/paper_table4_calibration_split_baselines.csv",
    diagnostic_table: str | Path = "reports/tables/paper_table5_diagnostic_baselines.csv",
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 20260428,
) -> Path:
    runs = load_paper_runs(
        suite=suite,
        run_dirs=[run_dir] if run_dir else None,
        run_root=Path(artifact_root) / "__no_reports_runs__" if artifact_root is not None and run_dir is None else "reports/runs",
        artifact_root=artifact_root,
        include_artifacts=include_artifacts,
    )
    if not runs:
        raise FileNotFoundError(f"No run artifacts found for suite={suite!r} run_dir={run_dir!r}")
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_suite = suite if suite is not None else run.suite or ""
        outcomes = outcomes_by_task(run.outcomes)
        metadata = _merged_task_metadata(run)
        prompt_by_task = _load_prompts(run.config.get("task_file"))
        output_tokens_by_task = _output_tokens_by_task(run.outcomes)
        task_records = _load_task_records(run.config.get("task_file"), metadata, prompt_by_task)
        if use_calibration_split:
            split_map = _load_split_map(Path(split_dir), run_suite)
            eval_task_ids = sorted(task_id for task_id, split in split_map.items() if split == "evaluation" and task_id in outcomes)
            calibration_task_ids = sorted(task_id for task_id, split in split_map.items() if split == "calibration" and task_id in outcomes)
            if not eval_task_ids or not calibration_task_ids:
                continue
            rows.extend(
                _calibration_split_rows(
                    run=run,
                    run_suite=run_suite,
                    eval_task_ids=eval_task_ids,
                    calibration_task_ids=calibration_task_ids,
                    outcomes=outcomes,
                    task_records=task_records,
                    include_test_distribution_diagnostics=include_test_distribution_diagnostics,
                    output_tokens_by_task=output_tokens_by_task,
                    n_bootstrap=n_bootstrap if bootstrap else 0,
                    bootstrap_seed=bootstrap_seed,
                )
            )
            continue
        groups = _groups(metadata, set(outcomes))
        for (track, source), task_ids in groups.items():
            baseline_curves = _baseline_curves(
                task_ids=task_ids,
                track=track,
                outcomes=outcomes,
                prompt_by_task=prompt_by_task,
                output_tokens_by_task=output_tokens_by_task,
                self_curves=forecast_curves(run.forecasts),
            )
            for method, curves in baseline_curves.items():
                scored = score_curve_set(curves, {task_id: outcomes[task_id] for task_id in curves})
                rows.append(
                    {
                        "suite": run_suite,
                        "run_id": run.run_id,
                        "model": run.model,
                        "forecast_method": method,
                        "baseline_class": _class_for_method(method),
                        "track": track,
                        "source": source,
                        "n_tasks": scored["n_tasks"],
                        "brier": scored["brier"],
                        "ece": scored["ece"],
                        "regret": scored["regret"],
                        "notes": _notes_for_method(method),
                    }
                )
    output_table = Path(output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_table, rows)
    _write_csv(Path(output_summary), _summary_rows(rows))
    calibration_rows = [row for row in rows if row.get("baseline_class") in {"model_forecast_raw", "model_forecast_recalibrated", "calibration_split_baseline"}]
    diagnostic_rows = [row for row in rows if row.get("baseline_class") in {"test_distribution_diagnostic", "posthoc_diagnostic"}]
    if use_calibration_split:
        _write_csv(Path(calibration_table), _paper_baseline_rows(calibration_rows))
        _write_csv(Path(diagnostic_table), _paper_baseline_rows(diagnostic_rows))
    _plot_baselines(rows, Path(output_figure_prefix))
    return output_table


def _calibration_split_rows(
    *,
    run,
    run_suite: str,
    eval_task_ids: list[str],
    calibration_task_ids: list[str],
    outcomes: dict[str, dict[int, bool]],
    task_records: dict[str, TaskRecord],
    include_test_distribution_diagnostics: bool,
    output_tokens_by_task: dict[str, int],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    self_curves = forecast_curves(run.forecasts)
    calibration_outcome_rows = [row for row in run.outcomes if str(row.get("task_id")) in set(calibration_task_ids)]
    calibration_forecasts = [row for row in run.forecasts if str(row.get("task_id")) in set(calibration_task_ids)]
    eval_forecasts = [row for row in run.forecasts if str(row.get("task_id")) in set(eval_task_ids)]
    budget_grid_by_task = {task_id: sorted(outcomes[task_id]) for task_id in eval_task_ids if task_id in outcomes}
    all_budgets = sorted({budget for task_id in eval_task_ids for budget in outcomes.get(task_id, {})})
    rows: list[dict[str, Any]] = []

    curves_by_method: dict[str, tuple[str, dict[str, dict[int, float]], str]] = {}
    curves_by_method["self_forecast_raw"] = (
        "model_forecast_raw",
        {task_id: self_curves[task_id] for task_id in eval_task_ids if task_id in self_curves},
        "raw model pre-execution forecast on heldout evaluation tasks",
    )
    calibrator = fit_histogram_recalibrator(calibration_forecasts, calibration_outcome_rows)
    curves_by_method["self_forecast_histogram_recalibrated"] = (
        "model_forecast_recalibrated",
        apply_histogram_recalibrator(eval_forecasts, calibrator),
        "histogram recalibration fit only on calibration tasks",
    )
    constant_fit = fit_constant_by_budget(calibration_outcome_rows)
    curves_by_method["constant_by_budget_calibration"] = (
        "calibration_split_baseline",
        predict_constant_by_budget(eval_task_ids, all_budgets, constant_fit),
        "empirical success by budget fit on calibration tasks",
    )
    source_fit = fit_source_by_budget(calibration_outcome_rows, task_records)
    curves_by_method["source_by_budget_calibration"] = (
        "calibration_split_baseline",
        predict_source_by_budget(eval_task_ids, task_records, budget_grid_by_task, source_fit),
        "source and budget prior fit on calibration tasks with global fallback",
    )
    prompt_fit = fit_prompt_length_bins(calibration_outcome_rows, task_records)
    curves_by_method["prompt_length_bin_calibration"] = (
        "calibration_split_baseline",
        predict_prompt_length_bins(eval_task_ids, task_records, budget_grid_by_task, prompt_fit),
        "prompt-length quantile-bin prior fit on calibration tasks",
    )
    curves_by_method["single_budget_midpoint"] = (
        "calibration_split_baseline",
        {
            task_id: {int(k): v for k, v in single_budget_curve(budget_grid_by_task[task_id][len(budget_grid_by_task[task_id]) // 2], budget_grid_by_task[task_id]).items()}
            for task_id in eval_task_ids
            if budget_grid_by_task.get(task_id)
        },
        "pre-execution threshold curve at the middle budget",
    )
    if include_test_distribution_diagnostics:
        curves_by_method["test_distribution_leave_one_out_source"] = (
            "test_distribution_diagnostic",
            _source_empirical_curves(eval_task_ids, outcomes),
            "leave-one-task-out prior on the evaluation distribution; diagnostic only",
        )
        curves_by_method["output_length_proxy_posthoc"] = (
            "posthoc_diagnostic",
            {
                task_id: {
                    int(k): v
                    for k, v in output_length_baseline_curve(
                        output_tokens_by_task.get(task_id) or budget_grid_by_task[task_id][len(budget_grid_by_task[task_id]) // 2],
                        budget_grid_by_task[task_id],
                    ).items()
                }
                for task_id in eval_task_ids
                if budget_grid_by_task.get(task_id)
            },
            "post-hoc output-token proxy; not deployable before spending tokens",
        )

    eval_outcomes = {task_id: outcomes[task_id] for task_id in eval_task_ids if task_id in outcomes}
    filtered_curves_by_method: dict[str, dict[str, dict[int, float]]] = {}
    metadata_by_method: dict[str, tuple[str, str]] = {}
    for method, (baseline_class, curves, notes) in curves_by_method.items():
        curves = {task_id: curve for task_id, curve in curves.items() if task_id in eval_outcomes}
        if curves:
            filtered_curves_by_method[method] = curves
            metadata_by_method[method] = (baseline_class, notes)
    bootstrap = bootstrap_baseline_metrics(
        filtered_curves_by_method,
        eval_outcomes,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    for method, (baseline_class, curves, notes) in curves_by_method.items():
        curves = filtered_curves_by_method.get(method, {})
        if not curves:
            continue
        scored = score_curve_set(curves, {task_id: eval_outcomes[task_id] for task_id in curves}, include_pairwise=False)
        intervals = bootstrap.get(method, {})
        rows.append(
            {
                "suite": run_suite,
                "run_id": run.run_id,
                "model": run.model,
                "forecast_method": method,
                "baseline_class": metadata_by_method.get(method, (baseline_class, notes))[0],
                "track": "all",
                "source": "all",
                "n_tasks": scored["n_tasks"],
                "n_eval_tasks": scored["n_tasks"],
                "n_calibration_tasks": len(calibration_task_ids),
                "calibration_task_ids_in_eval": len(set(calibration_task_ids) & set(curves)),
                "brier": scored["brier"],
                "ece": scored["ece"],
                "regret": scored["regret"],
                "brier_ci": intervals.get("brier", {}).get("ci", _point_or_blank(scored.get("brier"))),
                "ece_ci": intervals.get("ece", {}).get("ci", _point_or_blank(scored.get("ece"))),
                "regret_ci": intervals.get("regret", {}).get("ci", _point_or_blank(scored.get("regret"))),
                "brier_ci_low": intervals.get("brier", {}).get("ci_low", ""),
                "brier_ci_high": intervals.get("brier", {}).get("ci_high", ""),
                "ece_ci_low": intervals.get("ece", {}).get("ci_low", ""),
                "ece_ci_high": intervals.get("ece", {}).get("ci_high", ""),
                "regret_ci_low": intervals.get("regret", {}).get("ci_low", ""),
                "regret_ci_high": intervals.get("regret", {}).get("ci_high", ""),
                "n_bootstrap": n_bootstrap,
                "notes": metadata_by_method.get(method, (baseline_class, notes))[1],
            }
        )
    return rows


def bootstrap_baseline_metrics(
    curves_by_method: dict[str, dict[str, dict[int, float]]],
    outcomes: dict[str, dict[int, bool]],
    n_bootstrap: int = 1000,
    seed: int = 20260428,
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    """Bootstrap baseline metrics by evaluation task ID.

    Each resampled task keeps all of its budget rows. Duplicate sampled tasks
    are given unique synthetic keys so dict-based scoring preserves bootstrap
    multiplicity.
    """
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, dict[str, float | str | None]]] = {}
    for method, curves in sorted(curves_by_method.items()):
        task_ids = sorted(set(curves) & set(outcomes))
        if not task_ids:
            result[method] = {}
            continue
        base = score_curve_set(
            {task_id: curves[task_id] for task_id in task_ids},
            {task_id: outcomes[task_id] for task_id in task_ids},
            include_pairwise=False,
        )
        boot_values: dict[str, list[float]] = defaultdict(list)
        for iteration in range(max(0, int(n_bootstrap))):
            sample = rng.choice(task_ids, size=len(task_ids), replace=True).tolist()
            sample_curves: dict[str, dict[int, float]] = {}
            sample_outcomes: dict[str, dict[int, bool]] = {}
            for index, task_id in enumerate(sample):
                key = f"{task_id}__bootstrap_{index}"
                sample_curves[key] = curves[task_id]
                sample_outcomes[key] = outcomes[task_id]
            scored = score_curve_set(sample_curves, sample_outcomes, include_pairwise=False, ranking_seed=seed + iteration)
            for metric in ("brier", "ece", "regret"):
                value = scored.get(metric)
                if value is not None and np.isfinite(float(value)):
                    boot_values[metric].append(float(value))
        method_result: dict[str, dict[str, float | str | None]] = {}
        for metric in ("brier", "ece", "regret"):
            values = boot_values.get(metric, [])
            if values:
                _, low, high = metric_ci(values, 0.95)
            else:
                low = high = None
            estimate = base.get(metric)
            method_result[metric] = {
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "ci": ci_string(estimate, low, high) or _point_or_blank(estimate),
            }
        result[method] = method_result
    return result


def _groups(metadata: dict[str, dict[str, Any]], task_ids: set[str]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = {("all", "all"): sorted(task_ids)}
    for task_id in sorted(task_ids):
        meta = metadata.get(task_id, {})
        groups.setdefault((str(meta.get("track") or "unknown"), str(meta.get("source") or "unknown")), []).append(task_id)
    return groups


def _merged_task_metadata(run) -> dict[str, dict[str, Any]]:
    metadata = task_metadata(list(run.forecasts) + list(run.outcomes))
    for task_id, task in _load_task_records(run.config.get("task_file"), metadata, _load_prompts(run.config.get("task_file"))).items():
        metadata.setdefault(task_id, {}).update(
            {
                "track": task.track,
                "source": task.source,
                "source_version": task.source_version,
                "external_id": task.external_id,
                "prompt": task.prompt,
            }
        )
    return metadata


def _load_split_map(split_dir: Path, suite: str) -> dict[str, str]:
    path = split_dir / f"{suite}_calibration_eval_split.json"
    if not path.exists():
        raise FileNotFoundError(f"Calibration/evaluation split file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_splits = payload.get("task_splits")
    if not isinstance(task_splits, dict):
        raise ValueError(f"Split file missing task_splits object: {path}")
    return {str(task_id): str(split) for task_id, split in task_splits.items()}


def _load_task_records(
    task_file: str | None,
    metadata: dict[str, dict[str, Any]],
    prompt_by_task: dict[str, str],
) -> dict[str, TaskRecord]:
    records: dict[str, TaskRecord] = {}
    if task_file and Path(task_file).exists():
        for row in read_jsonl(task_file):
            task = TaskRecord.model_validate(row)
            records[task.task_id] = task
    for task_id, meta in metadata.items():
        if task_id in records:
            continue
        records[task_id] = TaskRecord(
            task_id=task_id,
            track=str(meta.get("track") or "math") if str(meta.get("track") or "math") in {"math", "coding", "swe", "agentic"} else "math",
            prompt=prompt_by_task.get(task_id, str(meta.get("prompt") or "")) or "unavailable prompt",
            verifier=str(meta.get("verifier") or "numeric_exact"),
            answer=None,
            source=str(meta.get("source") or "unknown"),
            source_version=str(meta.get("source_version") or meta.get("source") or "unknown"),
            external_id=str(meta.get("external_id") or task_id),
        )
    return records


def _baseline_curves(
    *,
    task_ids: list[str],
    track: str,
    outcomes: dict[str, dict[int, bool]],
    prompt_by_task: dict[str, str],
    output_tokens_by_task: dict[str, int],
    self_curves: dict[str, dict[int, float]],
) -> dict[str, dict[str, dict[int, float]]]:
    curves: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    constant_empirical = _constant_empirical_curves(task_ids, outcomes)
    source_empirical = _source_empirical_curves(task_ids, outcomes)
    for task_id in task_ids:
        if task_id not in outcomes:
            continue
        grid = sorted(outcomes[task_id])
        if task_id in self_curves:
            curves["self_forecast"][task_id] = self_curves[task_id]
        curves["constant_empirical_prior"][task_id] = constant_empirical.get(task_id) or {
            int(k): v for k, v in constant_curve(grid, probability=0.5).items()
        }
        curves["source_empirical_prior"][task_id] = source_empirical.get(task_id) or {
            int(k): v for k, v in domain_curve(track, grid).items()
        }
        curves["prompt_length_empirical"][task_id] = {
            int(k): v for k, v in prompt_length_curve(prompt_by_task.get(task_id, ""), grid).items()
        }
        predicted_output = output_tokens_by_task.get(task_id) or grid[len(grid) // 2]
        curves["output_length_proxy"][task_id] = {
            int(k): v for k, v in output_length_baseline_curve(predicted_output, grid).items()
        }
        curves["single_budget"][task_id] = {int(k): v for k, v in single_budget_curve(grid[len(grid) // 2], grid).items()}
    return dict(curves)


def _constant_empirical_curves(
    task_ids: list[str],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, dict[int, float]]:
    all_values = [
        bool(success)
        for task_id in task_ids
        for success in outcomes.get(task_id, {}).values()
    ]
    result: dict[str, dict[int, float]] = {}
    for task_id in task_ids:
        grid = sorted(outcomes.get(task_id, {}))
        task_values = [bool(success) for success in outcomes.get(task_id, {}).values()]
        denominator = len(all_values) - len(task_values)
        probability = (
            (sum(all_values) - sum(task_values)) / denominator
            if denominator > 0
            else (sum(all_values) / len(all_values) if all_values else 0.5)
        )
        result[task_id] = {budget: float(min(0.99, max(0.01, probability))) for budget in grid}
    return result


def _source_empirical_curves(
    task_ids: list[str],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, dict[int, float]]:
    by_budget: dict[int, list[tuple[str, bool]]] = defaultdict(list)
    for task_id in task_ids:
        for budget, success in outcomes.get(task_id, {}).items():
            by_budget[int(budget)].append((task_id, bool(success)))
    result: dict[str, dict[int, float]] = {}
    for task_id in task_ids:
        curve: dict[int, float] = {}
        for budget in sorted(outcomes.get(task_id, {})):
            peers = [success for peer_task, success in by_budget[int(budget)] if peer_task != task_id]
            if not peers:
                peers = [success for _, success in by_budget[int(budget)]]
            probability = sum(peers) / len(peers) if peers else 0.5
            curve[int(budget)] = float(min(0.99, max(0.01, probability)))
        result[task_id] = curve
    return result


def _load_prompts(task_file: str | None) -> dict[str, str]:
    if not task_file or not Path(task_file).exists():
        return {}
    return {str(row["task_id"]): str(row.get("prompt") or "") for row in read_jsonl(task_file)}


def _output_tokens_by_task(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in outcomes:
        if row.get("completion_tokens") is not None:
            grouped[str(row["task_id"])].append(int(row["completion_tokens"]))
    return {task_id: int(sum(values) / len(values)) for task_id, values in grouped.items() if values}


def _notes_for_method(method: str) -> str:
    return {
        "self_forecast": "model-provided pre-execution probability curve",
        "self_forecast_raw": "raw model pre-execution probability curve",
        "self_forecast_histogram_recalibrated": "histogram recalibration fit on calibration tasks",
        "constant_empirical_prior": "leave-one-task-out empirical success rate applied at every budget",
        "source_empirical_prior": "leave-one-task-out empirical success rate at the same budget within source group",
        "prompt_length_empirical": "metadata-only curve from prompt length",
        "output_length_proxy": "post-hoc output-token proxy; interpret as diagnostic",
        "output_length_proxy_posthoc": "post-hoc output-token proxy; interpret as diagnostic",
        "single_budget": "threshold curve at the middle budget",
        "single_budget_midpoint": "threshold curve at the middle budget",
        "constant_by_budget_calibration": "empirical success by budget fit on calibration tasks",
        "source_by_budget_calibration": "source and budget prior fit on calibration tasks",
        "prompt_length_bin_calibration": "prompt-length bin prior fit on calibration tasks",
        "test_distribution_leave_one_out_source": "leave-one-task-out evaluation-distribution diagnostic",
    }.get(method, "")


def _class_for_method(method: str) -> str:
    if method in {"output_length_proxy", "output_length_proxy_posthoc"}:
        return "posthoc_diagnostic"
    if method == "self_forecast":
        return "model_forecast_raw"
    if method == "source_empirical_prior":
        return "test_distribution_diagnostic"
    if method in {"constant_empirical_prior"}:
        return "test_distribution_diagnostic"
    return "calibration_split_baseline"


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("track") != "all" or row.get("source") != "all":
            continue
        grouped[(str(row.get("suite") or ""), row["forecast_method"], row["baseline_class"])].append(row)
    result: list[dict[str, Any]] = []
    for (suite, method, baseline_class), group in sorted(grouped.items()):
        result.append(
            {
                "suite": suite,
                "forecast_method": method,
                "baseline_class": baseline_class,
                "runs": len(group),
                "n_tasks": sum(int(row.get("n_tasks") or 0) for row in group),
                "n_eval_tasks": sum(int(row.get("n_eval_tasks") or row.get("n_tasks") or 0) for row in group),
                "mean_brier": _mean(_float_values(group, "brier")),
                "mean_ece": _mean(_float_values(group, "ece")),
                "mean_regret": _mean(_float_values(group, "regret")),
                "notes": _notes_for_method(method),
            }
        )
    return result


def _paper_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "suite": row.get("suite"),
                "model": row.get("model"),
                "forecast_method": row.get("forecast_method"),
                "baseline_class": row.get("baseline_class"),
                "n_eval_tasks": row.get("n_eval_tasks") or row.get("n_tasks"),
                "brier": row.get("brier"),
                "ece": row.get("ece"),
                "regret": row.get("regret"),
                "brier_low": row.get("brier_ci_low", ""),
                "brier_high": row.get("brier_ci_high", ""),
                "ece_low": row.get("ece_ci_low", ""),
                "ece_high": row.get("ece_ci_high", ""),
                "regret_low": row.get("regret_ci_low", ""),
                "regret_high": row.get("regret_ci_high", ""),
                "brier_ci": row.get("brier_ci") or _point_or_blank(row.get("brier")),
                "ece_ci": row.get("ece_ci") or _point_or_blank(row.get("ece")),
                "regret_ci": row.get("regret_ci") or _point_or_blank(row.get("regret")),
                "n_bootstrap": row.get("n_bootstrap", ""),
                "notes": row.get("notes"),
            }
        )
    return result


def _point_or_blank(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f} [point]"
    except Exception:
        return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_baselines(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        for suffix in ("png", "svg"):
            _empty_plot(output_prefix.with_suffix(f".{suffix}"), "No baseline rows")
        return
    rows = [row for row in rows if row.get("track") == "all" and row.get("source") == "all"]
    preferred = [
        "self_forecast_raw",
        "self_forecast_histogram_recalibrated",
        "constant_by_budget_calibration",
        "source_by_budget_calibration",
        "prompt_length_bin_calibration",
        "single_budget_midpoint",
        "test_distribution_leave_one_out_source",
        "output_length_proxy_posthoc",
        "self_forecast",
        "constant_empirical_prior",
        "source_empirical_prior",
        "prompt_length_empirical",
        "single_budget",
        "output_length_proxy",
    ]
    methods = [method for method in preferred if any(row["forecast_method"] == method for row in rows)]
    methods.extend(sorted({row["forecast_method"] for row in rows} - set(methods)))
    metrics = ["brier", "ece", "regret"]
    means = {
        metric: [
            _mean([float(row[metric]) for row in rows if row["forecast_method"] == method and row[metric] not in {None, ""}])
            for method in methods
        ]
        for metric in metrics
    }
    x = range(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    colors = ["#2563eb", "#059669", "#b45309"]
    for idx, metric in enumerate(metrics):
        ax.bar([i + (idx - 1) * width for i in x], means[metric], width=width, label=metric.upper(), color=colors[idx])
    ax.set_title("Forecast Baselines")
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(x), [method.replace("_", " ") for method in methods], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


def _empty_plot(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value not in {None, ""}:
            values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare self-forecasts against simple baselines.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--output-table", default="reports/tables/baseline_comparison.csv")
    parser.add_argument("--output-summary", default="reports/tables/baseline_summary.csv")
    parser.add_argument("--output-figure-prefix", default="reports/figures/figure_baseline_comparison")
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--use-calibration-split", action="store_true")
    parser.add_argument(
        "--include-test-distribution-diagnostics",
        action="store_true",
        default=True,
        help="Write heldout-distribution and post-hoc diagnostic baselines to the separate diagnostic table.",
    )
    parser.add_argument("--calibration-table", default="reports/tables/paper_table4_calibration_split_baselines.csv")
    parser.add_argument("--diagnostic-table", default="reports/tables/paper_table5_diagnostic_baselines.csv")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap deployable baseline metrics by heldout task ID.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", "--seed", dest="bootstrap_seed", type=int, default=20260428)
    args = parser.parse_args()
    path = run_baseline_analysis(
        suite=args.suite,
        run_dir=args.run_dir,
        artifact_root=args.artifact_root,
        include_artifacts=not args.no_artifacts,
        output_table=args.output_table,
        output_summary=args.output_summary,
        output_figure_prefix=args.output_figure_prefix,
        split_dir=args.split_dir,
        use_calibration_split=args.use_calibration_split,
        include_test_distribution_diagnostics=args.include_test_distribution_diagnostics,
        calibration_table=args.calibration_table,
        diagnostic_table=args.diagnostic_table,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
