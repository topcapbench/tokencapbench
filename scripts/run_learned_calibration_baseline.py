#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import forecast_curves, load_paper_runs, outcomes_by_task, score_curve_set, task_metadata
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
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl

try:
    from scripts.run_baseline_analysis import bootstrap_baseline_metrics
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_baseline_analysis import bootstrap_baseline_metrics


TABLE_FIELDS = [
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
]
METHOD_ORDER = [
    "self_forecast_raw",
    "self_forecast_histogram_recalibrated",
    "constant_by_budget_calibration",
    "source_by_budget_calibration",
    "prompt_length_bin_calibration",
    "learned_logistic_recalibrator",
]


def run_learned_calibration_baseline(
    *,
    artifact_root: str | Path = "reports/artifacts",
    split_dir: str | Path = "reports/splits",
    output_table: str | Path = "reports/tables/paper_table16_learned_calibration_baseline.csv",
    output_figure_prefix: str | Path = "reports/figures/paper_figure12_learned_calibration_baseline",
    n_bootstrap: int = 250,
    bootstrap_seed: int = 20260501,
    write_figure: bool = True,
    suites: Iterable[str] | None = None,
) -> list[Path]:
    requested_suites = [suite for suite in suites or [] if suite]
    if requested_suites:
        runs = []
        for suite in requested_suites:
            runs.extend(
                load_paper_runs(
                    suite=suite,
                    run_root=Path(artifact_root) / "__no_reports_runs__",
                    artifact_root=artifact_root,
                    include_artifacts=True,
                )
            )
    else:
        runs = load_paper_runs(
            run_root=Path(artifact_root) / "__no_reports_runs__",
            artifact_root=artifact_root,
            include_artifacts=True,
        )
    rows: list[dict[str, Any]] = []
    for run in runs:
        if str(run.model).lower().replace("_", "-") == "mock-model":
            continue
        suite = run.suite or ""
        outcomes = outcomes_by_task(run.outcomes)
        split_map = _load_split_map(Path(split_dir), suite)
        calibration_ids, eval_ids = _calibration_eval_ids(split_map, outcomes)
        if not calibration_ids or not eval_ids:
            continue
        task_records = _task_records_by_id(run)
        curves_by_method = _curves_by_method(
            forecasts=run.forecasts,
            outcomes=outcomes,
            calibration_ids=calibration_ids,
            eval_ids=eval_ids,
            task_records=task_records,
        )
        eval_outcomes = {task_id: outcomes[task_id] for task_id in eval_ids if task_id in outcomes}
        curves_by_method = {
            method: _filter_curves(curves, eval_outcomes)
            for method, curves in curves_by_method.items()
        }
        curves_by_method = {method: curves for method, curves in curves_by_method.items() if curves}
        if not curves_by_method:
            continue
        intervals = bootstrap_baseline_metrics(
            curves_by_method,
            eval_outcomes,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        for method in METHOD_ORDER:
            curves = curves_by_method.get(method)
            if not curves:
                continue
            scored = score_curve_set(
                curves,
                {task_id: eval_outcomes[task_id] for task_id in curves},
                include_pairwise=False,
            )
            method_intervals = intervals.get(method, {})
            rows.append(
                {
                    "suite": suite,
                    "model": run.model,
                    "method": method,
                    "n_calibration_tasks": len(calibration_ids),
                    "n_eval_tasks": scored.get("n_tasks", len(curves)),
                    "brier": _fmt(scored.get("brier")),
                    "ece": _fmt(scored.get("ece")),
                    "regret": _fmt(scored.get("regret")),
                    "brier_ci": method_intervals.get("brier", {}).get("ci", _point_ci(scored.get("brier"))),
                    "ece_ci": method_intervals.get("ece", {}).get("ci", _point_ci(scored.get("ece"))),
                    "regret_ci": method_intervals.get("regret", {}).get("ci", _point_ci(scored.get("regret"))),
                }
            )

    table_path = Path(output_table)
    _write_csv(table_path, rows)
    outputs = [table_path]
    if write_figure:
        outputs.extend(_plot_learned_baseline(rows, Path(output_figure_prefix)))
    return outputs


def plot_learned_calibration_baseline(
    *,
    table_path: str | Path = "reports/tables/paper_table16_learned_calibration_baseline.csv",
    output_figure_prefix: str | Path = "reports/figures/paper_figure12_learned_calibration_baseline",
) -> list[Path]:
    return _plot_learned_baseline(_read_csv(Path(table_path)), Path(output_figure_prefix))


def _curves_by_method(
    *,
    forecasts: list[dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
    calibration_ids: list[str],
    eval_ids: list[str],
    task_records: dict[str, TaskRecord],
) -> dict[str, dict[str, dict[int, float]]]:
    calibration_set = set(calibration_ids)
    eval_set = set(eval_ids)
    self_curves = forecast_curves(forecasts)
    calibration_outcomes = [
        {"task_id": task_id, "budget": budget, "success": success}
        for task_id in calibration_ids
        for budget, success in outcomes.get(task_id, {}).items()
    ]
    calibration_forecasts = [row for row in forecasts if str(row.get("task_id")) in calibration_set]
    eval_forecasts = [row for row in forecasts if str(row.get("task_id")) in eval_set]
    budget_grid_by_task = {task_id: sorted(outcomes[task_id]) for task_id in eval_ids if task_id in outcomes}
    all_budgets = sorted({budget for task_id in eval_ids for budget in outcomes.get(task_id, {})})
    curves: dict[str, dict[str, dict[int, float]]] = {
        "self_forecast_raw": {task_id: self_curves[task_id] for task_id in eval_ids if task_id in self_curves},
    }
    hist = fit_histogram_recalibrator(calibration_forecasts, calibration_outcomes)
    curves["self_forecast_histogram_recalibrated"] = apply_histogram_recalibrator(eval_forecasts, hist)
    constant_fit = fit_constant_by_budget(calibration_outcomes)
    curves["constant_by_budget_calibration"] = predict_constant_by_budget(eval_ids, all_budgets, constant_fit)
    source_fit = fit_source_by_budget(calibration_outcomes, task_records)
    curves["source_by_budget_calibration"] = predict_source_by_budget(eval_ids, task_records, budget_grid_by_task, source_fit)
    prompt_fit = fit_prompt_length_bins(calibration_outcomes, task_records)
    curves["prompt_length_bin_calibration"] = predict_prompt_length_bins(eval_ids, task_records, budget_grid_by_task, prompt_fit)
    curves["learned_logistic_recalibrator"] = learned_logistic_curves(
        raw_curves=self_curves,
        outcomes=outcomes,
        calibration_ids=calibration_ids,
        eval_ids=eval_ids,
        task_records=task_records,
    )
    return curves


def learned_logistic_curves(
    *,
    raw_curves: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
    calibration_ids: list[str],
    eval_ids: list[str],
    task_records: dict[str, TaskRecord],
) -> dict[str, dict[int, float]]:
    encoder = _FeatureEncoder(task_records, raw_curves, outcomes)
    train_x: list[list[float]] = []
    train_y: list[bool] = []
    for task_id in calibration_ids:
        for budget, probability in raw_curves.get(task_id, {}).items():
            if budget in outcomes.get(task_id, {}):
                train_x.append(encoder.row(task_id, budget, probability))
                train_y.append(bool(outcomes[task_id][budget]))
    eval_items: list[tuple[str, int, float]] = []
    eval_x: list[list[float]] = []
    for task_id in eval_ids:
        for budget, probability in raw_curves.get(task_id, {}).items():
            if budget in outcomes.get(task_id, {}):
                eval_items.append((task_id, int(budget), float(probability)))
                eval_x.append(encoder.row(task_id, int(budget), float(probability)))
    if not train_x or not eval_x:
        return {}
    probabilities = fit_learned_recalibrator_predict(train_x, train_y, eval_x)
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (task_id, budget, _raw_probability), probability in zip(eval_items, probabilities):
        result[task_id][int(budget)] = float(max(0.0, min(1.0, probability)))
    return {task_id: dict(sorted(curve.items())) for task_id, curve in result.items()}


def fit_learned_recalibrator_predict(
    train_x: list[list[float]],
    train_y: list[bool],
    eval_x: list[list[float]],
) -> list[float]:
    y = np.asarray([1.0 if value else 0.0 for value in train_y], dtype=float)
    if len(set(train_y)) < 2:
        return [float(np.mean(y)) for _ in eval_x]
    x_train = np.asarray(train_x, dtype=float)
    x_eval = np.asarray(eval_x, dtype=float)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        model.fit(x_train, y.astype(int))
        return [float(value) for value in model.predict_proba(x_eval)[:, 1]]
    except Exception:
        return _numpy_logistic_predict(x_train, y, x_eval)


def _numpy_logistic_predict(x_train: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> list[float]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (x_train - mean) / scale
    eval_matrix = (x_eval - mean) / scale
    train = np.column_stack([np.ones(train.shape[0]), train])
    eval_matrix = np.column_stack([np.ones(eval_matrix.shape[0]), eval_matrix])
    weights = np.zeros(train.shape[1], dtype=float)
    lr = 0.08
    l2 = 1.0
    for _ in range(1400):
        logits = np.clip(train @ weights, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))
        gradient = (train.T @ (probs - y)) / len(y)
        gradient[1:] += (1.0 / l2) * weights[1:] / len(y)
        weights -= lr * gradient
    pred = 1.0 / (1.0 + np.exp(-np.clip(eval_matrix @ weights, -30, 30)))
    return [float(value) for value in pred]


class _FeatureEncoder:
    def __init__(
        self,
        task_records: dict[str, TaskRecord],
        raw_curves: dict[str, dict[int, float]],
        outcomes: dict[str, dict[int, bool]],
    ) -> None:
        self.task_records = task_records
        self.sources = sorted({task.source for task in task_records.values()}) or ["unknown"]
        lengths = [len(task.prompt or "") for task in task_records.values()]
        self.prompt_cutoffs = np.quantile(lengths, [0.25, 0.5, 0.75]).tolist() if lengths else [0, 0, 0]
        self.budget_positions: dict[tuple[str, int], float] = {}
        for task_id in set(raw_curves) | set(outcomes):
            grid = sorted(set(raw_curves.get(task_id, {})) | set(outcomes.get(task_id, {})))
            denominator = max(1, len(grid) - 1)
            for index, budget in enumerate(grid):
                self.budget_positions[(task_id, int(budget))] = index / denominator

    def row(self, task_id: str, budget: int, raw_probability: float) -> list[float]:
        task = self.task_records.get(task_id)
        source = task.source if task else "unknown"
        prompt_length = len(task.prompt or "") if task else 0
        prompt_bin = sum(1 for cutoff in self.prompt_cutoffs if prompt_length > cutoff)
        features = [
            float(max(0.0, min(1.0, raw_probability))),
            math.log(max(1, int(budget))),
            self.budget_positions.get((task_id, int(budget)), 0.0),
        ]
        features.extend(1.0 if source == value else 0.0 for value in self.sources)
        features.extend(1.0 if prompt_bin == index else 0.0 for index in range(4))
        return features


def _calibration_eval_ids(split_map: dict[str, str], outcomes: dict[str, dict[int, bool]]) -> tuple[list[str], list[str]]:
    calibration_ids = sorted(task_id for task_id, split in split_map.items() if split == "calibration" and task_id in outcomes)
    eval_ids = sorted(task_id for task_id, split in split_map.items() if split == "evaluation" and task_id in outcomes)
    if set(calibration_ids) & set(eval_ids):
        raise ValueError("Calibration and evaluation task IDs overlap.")
    return calibration_ids, eval_ids


def _load_split_map(split_dir: Path, suite: str) -> dict[str, str]:
    path = split_dir / f"{suite}_calibration_eval_split.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_splits = payload.get("task_splits")
    return {str(task_id): str(split) for task_id, split in task_splits.items()} if isinstance(task_splits, dict) else {}


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
                    "verifier": task.verifier,
                }
            )
    result: dict[str, TaskRecord] = {}
    for task_id, meta in metadata.items():
        track = str(meta.get("track") or "math")
        if track not in {"math", "coding", "swe", "agentic"}:
            track = "math"
        result[task_id] = TaskRecord(
            task_id=task_id,
            track=track,
            prompt=str(meta.get("prompt") or "unavailable prompt"),
            verifier=str(meta.get("verifier") or "numeric_exact"),
            answer=None,
            source=str(meta.get("source") or "unknown"),
            source_version=str(meta.get("source_version") or meta.get("source") or "unknown"),
            external_id=str(meta.get("external_id") or task_id),
        )
    return result


def _filter_curves(
    curves: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, dict[int, float]]:
    filtered: dict[str, dict[int, float]] = {}
    for task_id, curve in curves.items():
        task_outcomes = outcomes.get(task_id)
        if not task_outcomes:
            continue
        allowed = {
            int(budget): float(max(0.0, min(1.0, probability)))
            for budget, probability in curve.items()
            if int(budget) in task_outcomes
        }
        if allowed:
            filtered[task_id] = dict(sorted(allowed.items()))
    return filtered


def _plot_learned_baseline(rows: list[dict[str, Any]], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    if not rows:
        axes[0].text(0.5, 0.5, "No learned calibration rows", ha="center", va="center", transform=axes[0].transAxes)
        for ax in axes:
            ax.axis("off")
    else:
        methods = [method for method in METHOD_ORDER if any(row.get("method") == method for row in rows)]
        labels = [_label(method) for method in methods]
        colors = [_color(method) for method in methods]
        for ax, metric in zip(axes, ["brier", "ece", "regret"]):
            values = [
                _mean_metric(
                    [row for row in rows if row.get("method") == method and row.get(metric) not in {"", None}],
                    metric,
                )
                for method in methods
            ]
            ax.bar(np.arange(len(methods)), values, color=colors, alpha=0.9)
            ax.set_xticks(np.arange(len(methods)), labels, rotation=30, ha="right")
            ax.set_ylabel(metric.upper() if metric != "ece" else "ECE")
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.grid(axis="y", alpha=0.35)
    fig.suptitle("Learned calibration baseline against deployable alternatives", fontsize=13.2, fontweight="bold")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return paths


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if row.get(metric) not in {"", None}]
    return sum(values) / len(values) if values else 0.0


def _label(method: str) -> str:
    return {
        "self_forecast_raw": "Raw",
        "self_forecast_histogram_recalibrated": "Histogram",
        "constant_by_budget_calibration": "Budget prior",
        "source_by_budget_calibration": "Source prior",
        "prompt_length_bin_calibration": "Prompt prior",
        "learned_logistic_recalibrator": "Learned",
    }.get(method, method.replace("_", " "))


def _color(method: str) -> str:
    return {
        "self_forecast_raw": "#2563eb",
        "self_forecast_histogram_recalibrated": "#059669",
        "constant_by_budget_calibration": "#92400e",
        "source_by_budget_calibration": "#b45309",
        "prompt_length_bin_calibration": "#d97706",
        "learned_logistic_recalibrator": "#7c3aed",
    }.get(method, "#4b5563")


def _fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value):.6f}"


def _point_ci(value: Any) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value):.3f} [point]"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in TABLE_FIELDS} for row in rows])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate a calibration-split learned logistic recalibrator.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--output-table", default="reports/tables/paper_table16_learned_calibration_baseline.csv")
    parser.add_argument("--output-figure-prefix", default="reports/figures/paper_figure12_learned_calibration_baseline")
    parser.add_argument("--n-bootstrap", type=int, default=250)
    parser.add_argument("--bootstrap-seed", type=int, default=20260501)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--suite", action="append", default=[], help="Suite to include; repeatable.")
    args = parser.parse_args()
    outputs = run_learned_calibration_baseline(
        artifact_root=args.artifact_root,
        split_dir=args.split_dir,
        output_table=args.output_table,
        output_figure_prefix=args.output_figure_prefix,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        write_figure=not args.no_figure,
        suites=args.suite,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
