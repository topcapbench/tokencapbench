#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import forecast_curves, load_paper_runs, outcomes_by_task
from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.metrics.first_success_budget import observed_first_success_budget
from budget2success.utils.jsonl import read_jsonl


TABLE_FIELDS = [
    "suite",
    "model",
    "n_eval_tasks",
    "n_calibration_tasks",
    "calibration_task_ids_in_eval",
    "n_with_usage_forecast",
    "usage_forecast_coverage",
    "corr_predicted_total_visible_tokens_to_observed_first_success_budget",
    "corr_predicted_unconstrained_output_tokens_to_observed_first_success_budget",
    "success_forecast_ranking_accuracy",
    "token_usage_proxy_ranking_accuracy",
    "raw_success_brier",
    "raw_success_ece",
    "token_proxy_calibrated_brier",
    "token_proxy_calibrated_ece",
    "notes",
]


def analyze_token_usage_proxy(
    *,
    artifact_root: str | Path | list[str | Path] | tuple[str | Path, ...] = "reports/artifacts",
    split_dir: str | Path = "reports/splits",
    dual_forecast_root: str | Path = "reports/runs/paper_dual_success_usage_forecast",
    output_table: str | Path = "reports/tables/paper_table13_token_usage_proxy.csv",
    output_figure_prefix: str | Path = "reports/figures/paper_figure10_token_usage_proxy_vs_success",
    write_figure: bool = True,
    suite_filter: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    dual_forecasts = _load_dual_forecasts(Path(dual_forecast_root))
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
    rows: list[dict[str, Any]] = []
    scatter_points: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for run in runs:
        suite = run.suite or ""
        if allowed_suites is not None and suite not in allowed_suites:
            continue
        outcomes = outcomes_by_task(run.outcomes)
        split_map = _load_split_map(Path(split_dir), suite)
        eval_ids = [task_id for task_id, split in split_map.items() if split == "evaluation" and task_id in outcomes]
        calibration_ids = [task_id for task_id, split in split_map.items() if split == "calibration" and task_id in outcomes]
        if not eval_ids:
            eval_ids = sorted(outcomes)
        eval_set = set(eval_ids)
        calibration_set = set(calibration_ids)
        model_dual = dual_forecasts.get(run.model, {})
        forecast_by_task = {
            str(row.get("task_id")): row
            for row in run.forecasts
            if row.get("task_id") is not None and row.get("p_success_by_budget")
        }
        forecast_by_task.update(
            {
                task_id: row
                for task_id, row in model_dual.items()
                if task_id in outcomes and (not row.get("metadata") or _source_to_suite(row.get("metadata", {}).get("source")) in {"", suite})
            }
        )
        scored = _score_run(
            suite=suite,
            model=run.model,
            forecasts=forecast_by_task,
            outcomes=outcomes,
            eval_ids=eval_ids,
            calibration_ids=calibration_ids,
        )
        rows.append(scored["row"])
        scatter_points.extend(scored["scatter_points"])
        ranking_rows.append(scored["ranking_row"])

    table_path = Path(output_table)
    _write_csv(table_path, rows)
    figure_paths = _plot_proxy_figure(scatter_points, ranking_rows, Path(output_figure_prefix)) if write_figure else []
    return [table_path, *figure_paths]


def _score_run(
    *,
    suite: str,
    model: str,
    forecasts: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
    eval_ids: list[str],
    calibration_ids: list[str],
) -> dict[str, Any]:
    eval_set = set(eval_ids)
    calibration_set = set(calibration_ids)
    overlap = len(eval_set & calibration_set)
    eval_forecasts = {task_id: forecasts[task_id] for task_id in eval_ids if task_id in forecasts and task_id in outcomes}
    curves = forecast_curves(eval_forecasts.values())
    probs, labels = _probability_label_pairs(curves, {task_id: outcomes[task_id] for task_id in eval_forecasts})
    raw_brier = _finite(brier_score(probs, labels)) if probs else None
    raw_ece = _finite(expected_calibration_error(probs, labels)) if probs else None

    predicted_success_need: dict[str, float | None] = {}
    predicted_usage_need: dict[str, float | None] = {}
    observed_need: dict[str, int | None] = {}
    scatter_points: list[dict[str, Any]] = []
    for task_id, forecast in eval_forecasts.items():
        observed = observed_first_success_budget(outcomes.get(task_id, {}))
        observed_need[task_id] = observed
        predicted_success_need[task_id] = _predicted_first_success_budget(forecast)
        usage = _usage_value(
            forecast,
            "predicted_total_visible_tokens_to_solve",
            aliases=(
                "predicted_total_visible_tokens",
                "predicted_visible_tokens_unconstrained",
                "predicted_unconstrained_output_tokens",
                "predicted_output_tokens_unconstrained",
            ),
        )
        predicted_usage_need[task_id] = usage
        if usage is not None and observed is not None:
            scatter_points.append(
                {
                    "suite": suite,
                    "model": model,
                    "predicted_usage": float(usage),
                    "observed_first_success_budget": int(observed),
                }
            )

    total_visible = [
        (
            _usage_value(
                forecast,
                "predicted_total_visible_tokens_to_solve",
                aliases=(
                    "predicted_total_visible_tokens",
                    "predicted_visible_tokens_unconstrained",
                    "predicted_unconstrained_output_tokens",
                    "predicted_output_tokens_unconstrained",
                ),
            ),
            observed_first_success_budget(outcomes[task_id]),
        )
        for task_id, forecast in eval_forecasts.items()
        if task_id in outcomes
    ]
    unconstrained = [
        (
            _usage_value(
                forecast,
                "predicted_unconstrained_output_tokens",
                aliases=("predicted_output_tokens_unconstrained",),
            ),
            observed_first_success_budget(outcomes[task_id]),
        )
        for task_id, forecast in eval_forecasts.items()
        if task_id in outcomes
    ]
    n_with_usage = sum(1 for value, _observed in total_visible if value is not None)
    proxy_scores = _calibrated_token_proxy_scores(
        forecasts=forecasts,
        outcomes=outcomes,
        calibration_ids=calibration_ids,
        eval_ids=eval_ids,
    )
    proxy_probs = []
    proxy_labels = []
    for task_id, by_budget in proxy_scores.items():
        for budget, probability in by_budget.items():
            if budget in outcomes.get(task_id, {}):
                proxy_probs.append(probability)
                proxy_labels.append(bool(outcomes[task_id][budget]))

    row = {
        "suite": suite,
        "model": model,
        "n_eval_tasks": len(eval_forecasts),
        "n_calibration_tasks": len(calibration_ids),
        "calibration_task_ids_in_eval": overlap,
        "n_with_usage_forecast": n_with_usage,
        "usage_forecast_coverage": _fmt(n_with_usage / len(eval_forecasts) if eval_forecasts else 0.0),
        "corr_predicted_total_visible_tokens_to_observed_first_success_budget": _fmt_or_na(_pearson(total_visible)),
        "corr_predicted_unconstrained_output_tokens_to_observed_first_success_budget": _fmt_or_na(_pearson(unconstrained)),
        "success_forecast_ranking_accuracy": _fmt_or_na(_ranking_accuracy(predicted_success_need, observed_need)),
        "token_usage_proxy_ranking_accuracy": _fmt_or_na(_ranking_accuracy(predicted_usage_need, observed_need)),
        "raw_success_brier": _fmt_or_na(raw_brier),
        "raw_success_ece": _fmt_or_na(raw_ece),
        "token_proxy_calibrated_brier": _fmt_or_na(_finite(brier_score(proxy_probs, proxy_labels)) if proxy_probs else None),
        "token_proxy_calibrated_ece": _fmt_or_na(_finite(expected_calibration_error(proxy_probs, proxy_labels)) if proxy_probs else None),
        "notes": (
            "token-use proxy uses total-visible forecast when present, otherwise unconstrained-output forecast"
            if n_with_usage
            else "no preserved token-usage forecast extras found; proxy metrics marked NA"
        ),
    }
    ranking_row = {
        "suite": suite,
        "model": model,
        "success_forecast_ranking_accuracy": row["success_forecast_ranking_accuracy"],
        "token_usage_proxy_ranking_accuracy": row["token_usage_proxy_ranking_accuracy"],
    }
    return {"row": row, "scatter_points": scatter_points, "ranking_row": ranking_row}


def _calibrated_token_proxy_scores(
    *,
    forecasts: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
    calibration_ids: list[str],
    eval_ids: list[str],
) -> dict[str, dict[int, float]]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for task_id in calibration_ids:
        forecast = forecasts.get(task_id)
        usage = (
            _usage_value(
                forecast,
                "predicted_total_visible_tokens_to_solve",
                aliases=(
                    "predicted_total_visible_tokens",
                    "predicted_visible_tokens_unconstrained",
                    "predicted_unconstrained_output_tokens",
                    "predicted_output_tokens_unconstrained",
                ),
            )
            if forecast
            else None
        )
        if usage is None or task_id not in outcomes:
            continue
        for budget, success in outcomes[task_id].items():
            grouped[int(budget >= usage)].append(bool(success))
    if not grouped:
        return {}
    defaults = {bucket: (sum(values) / len(values) if values else 0.5) for bucket, values in grouped.items()}
    global_default = sum(success for values in grouped.values() for success in values) / sum(len(values) for values in grouped.values())
    result: dict[str, dict[int, float]] = {}
    for task_id in eval_ids:
        forecast = forecasts.get(task_id)
        usage = (
            _usage_value(
                forecast,
                "predicted_total_visible_tokens_to_solve",
                aliases=(
                    "predicted_total_visible_tokens",
                    "predicted_visible_tokens_unconstrained",
                    "predicted_unconstrained_output_tokens",
                    "predicted_output_tokens_unconstrained",
                ),
            )
            if forecast
            else None
        )
        if usage is None or task_id not in outcomes:
            continue
        result[task_id] = {
            int(budget): float(defaults.get(int(budget >= usage), global_default))
            for budget in sorted(outcomes[task_id])
        }
    return result


def _plot_proxy_figure(scatter_points: list[dict[str, Any]], ranking_rows: list[dict[str, Any]], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    if scatter_points:
        suite_colors = {
            "paper_bigcodebench_hard": "#2563eb",
            "paper_canitedit_descriptive": "#059669",
            "paper_evalplus_humaneval_full": "#7c3aed",
            "paper_evalplus_mbpp_full": "#dc2626",
            "paper_math_core": "#d97706",
        }
        for suite in sorted({str(point.get("suite") or "") for point in scatter_points}):
            points = [point for point in scatter_points if str(point.get("suite") or "") == suite]
            xs = [point["predicted_usage"] for point in points]
            ys = [point["observed_first_success_budget"] for point in points]
            ax.scatter(
                xs,
                ys,
                s=22,
                alpha=0.62,
                color=suite_colors.get(suite, "#4b5563"),
                edgecolors="white",
                linewidths=0.25,
                label=_suite_label(suite),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Predicted token-use proxy")
        ax.set_ylabel("Observed first-success budget")
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    else:
        ax.text(
            0.5,
            0.5,
            "No paired usage and first-success observations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    ax.set_title("Usage Proxy vs Observed Need", fontweight="bold")

    ax = axes[1]
    success_values = [_float_or_none(row["success_forecast_ranking_accuracy"]) for row in ranking_rows]
    proxy_values = [_float_or_none(row["token_usage_proxy_ranking_accuracy"]) for row in ranking_rows]
    success_values = [value for value in success_values if value is not None]
    proxy_values = [value for value in proxy_values if value is not None]
    labels = ["Success forecast", "Token-usage proxy"]
    values = [
        sum(success_values) / len(success_values) if success_values else 0.0,
        sum(proxy_values) / len(proxy_values) if proxy_values else 0.0,
    ]
    colors = ["#059669", "#dc2626"]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.98, value + 0.025),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairwise ranking accuracy")
    ax.grid(axis="y", alpha=0.35)
    if not proxy_values:
        ax.text(1, 0.08, "NA", ha="center", va="bottom", color="#7f1d1d", fontweight="bold")
    ax.set_title("Ranking Accuracy", fontweight="bold")
    fig.suptitle("Token-usage proxy versus success-under-budget forecasting", fontsize=13.2, fontweight="bold")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return paths


def _load_dual_forecasts(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if not root.exists():
        return {}
    for forecast_path in sorted(root.glob("*/forecasts.jsonl")):
        for row in read_jsonl(forecast_path):
            task_id = row.get("task_id")
            model = row.get("model")
            if task_id and model and row.get("p_success_by_budget"):
                result[str(model)][str(task_id)] = row
    return dict(result)


def _load_split_map(split_dir: Path, suite: str) -> dict[str, str]:
    path = split_dir / f"{suite}_calibration_eval_split.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_splits = payload.get("task_splits")
    return {str(task_id): str(split) for task_id, split in task_splits.items()} if isinstance(task_splits, dict) else {}


def _source_to_suite(source: Any) -> str:
    return {
        "gsm8k": "paper_math_core",
        "hendrycks_math": "paper_math_core",
        "evalplus_humaneval": "paper_evalplus_humaneval_full",
        "evalplus_mbpp": "paper_evalplus_mbpp_full",
        "bigcodebench_hard": "paper_bigcodebench_hard",
        "canitedit": "paper_canitedit_descriptive",
        "aider_polyglot": "paper_aider_polyglot",
    }.get(str(source or ""), "")


def _suite_label(suite: str) -> str:
    return {
        "paper_bigcodebench_hard": "BigCodeBench-Hard",
        "paper_canitedit_descriptive": "CanItEdit",
        "paper_evalplus_humaneval_full": "HumanEval+",
        "paper_evalplus_mbpp_full": "MBPP+",
        "paper_math_core": "Math",
    }.get(str(suite or ""), str(suite or "unknown").replace("_", " "))


def _probability_label_pairs(
    curves: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
) -> tuple[list[float], list[bool]]:
    probs: list[float] = []
    labels: list[bool] = []
    for task_id, curve in curves.items():
        for budget, probability in curve.items():
            if budget in outcomes.get(task_id, {}):
                probs.append(float(probability))
                labels.append(bool(outcomes[task_id][budget]))
    return probs, labels


def _predicted_first_success_budget(forecast: dict[str, Any]) -> float | None:
    direct = _usage_value(forecast, "predicted_first_success_budget")
    if direct is not None:
        return direct
    median = forecast.get("median_budget2success")
    if median is not None:
        try:
            return float(median)
        except (TypeError, ValueError):
            return None
    curve = forecast.get("p_success_by_budget") or {}
    candidates = [(int(budget), float(probability)) for budget, probability in curve.items()]
    if not candidates:
        return None
    for budget, probability in sorted(candidates):
        if probability >= 0.5:
            return float(budget)
    return float(max(budget for budget, _probability in candidates))


def _usage_value(forecast: dict[str, Any] | None, key: str, *, aliases: tuple[str, ...] = ()) -> float | None:
    if not forecast:
        return None
    containers = [forecast.get("forecast_extras") if isinstance(forecast.get("forecast_extras"), dict) else {}, forecast]
    keys = (key, *aliases)
    for container in containers:
        for field in keys:
            value = container.get(field)
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0:
                return number
    return None


def _pearson(pairs: list[tuple[float | None, int | None]]) -> float | None:
    values = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    if len(values) < 2:
        return None
    xs = np.array([x for x, _y in values], dtype=float)
    ys = np.array([y for _x, y in values], dtype=float)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _ranking_accuracy(predicted: dict[str, float | None], observed: dict[str, int | None]) -> float | None:
    task_ids = [task_id for task_id in sorted(predicted) if predicted[task_id] is not None and observed.get(task_id) is not None]
    total = 0
    score = 0.0
    for index, left in enumerate(task_ids):
        for right in task_ids[index + 1 :]:
            obs_left = observed[left]
            obs_right = observed[right]
            if obs_left == obs_right:
                continue
            total += 1
            pred_left = float(predicted[left])
            pred_right = float(predicted[right])
            if pred_left == pred_right:
                score += 0.5
            elif (pred_left < pred_right) == (obs_left < obs_right):
                score += 1.0
    return score / total if total else None


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _fmt_or_na(value: float | None) -> str:
    return "NA" if value is None else _fmt(value)


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _float_or_none(value: Any) -> float | None:
    if value in {None, "", "NA"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in TABLE_FIELDS} for row in rows])


def _artifact_roots(value: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]
    return [Path(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare token-usage proxy predictions with success-under-budget forecasts.")
    parser.add_argument("--artifact-root", action="append", default=None)
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--dual-forecast-root", default="reports/runs/paper_dual_success_usage_forecast")
    parser.add_argument("--output-table", default="reports/tables/paper_table13_token_usage_proxy.csv")
    parser.add_argument("--output-figure-prefix", default="reports/figures/paper_figure10_token_usage_proxy_vs_success")
    parser.add_argument("--suite", action="append", default=None, help="Restrict analysis to a suite; repeatable.")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    outputs = analyze_token_usage_proxy(
        artifact_root=args.artifact_root or ["reports/artifacts"],
        split_dir=args.split_dir,
        dual_forecast_root=args.dual_forecast_root,
        output_table=args.output_table,
        output_figure_prefix=args.output_figure_prefix,
        write_figure=not args.no_figure,
        suite_filter=set(args.suite) if args.suite else None,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
