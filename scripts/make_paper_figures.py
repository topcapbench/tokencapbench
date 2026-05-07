#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import (
    calibration_points,
    forecast_curves,
    forecast_medians,
    load_paper_runs,
    outcomes_by_task,
    probability_label_pairs,
    score_curve_set,
)
from budget2success.metrics.regret import oracle_utility, selected_budget_from_forecast, utility
from budget2success.metrics.first_success_budget import observed_censored_at_budget, observed_first_success_budget

try:
    from scripts.run_allocation_frontier import (
        plot_allocation_frontier,
        plot_fixed_budget_scheduling,
        plot_replacement_allocation_frontier,
        run_allocation_frontier,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_allocation_frontier import (
        plot_allocation_frontier,
        plot_fixed_budget_scheduling,
        plot_replacement_allocation_frontier,
        run_allocation_frontier,
    )
try:
    from scripts.analyze_token_usage_proxy import analyze_token_usage_proxy
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from analyze_token_usage_proxy import analyze_token_usage_proxy
try:
    from scripts.analyze_forecast_stability import analyze_forecast_stability
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from analyze_forecast_stability import analyze_forecast_stability
try:
    from scripts.run_learned_calibration_baseline import plot_learned_calibration_baseline, run_learned_calibration_baseline
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_learned_calibration_baseline import plot_learned_calibration_baseline, run_learned_calibration_baseline


FIGURE_DIR = Path("reports/figures")
MODEL_COLORS = {
    "DeepSeek-V3-0324": "#2563eb",
    "claude-3-haiku": "#7c3aed",
    "gemini-2.0-flash-001": "#059669",
    "gemini-2.0-flash-lite-001": "#dc2626",
    "gpt-5-mini": "#111827",
}
MODEL_MARKERS = {
    "DeepSeek-V3-0324": "o",
    "claude-3-haiku": "s",
    "gemini-2.0-flash-001": "D",
    "gemini-2.0-flash-lite-001": "^",
    "gpt-5-mini": "P",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 9,
            "legend.fontsize": 7.5,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def make_paper_figures(
    *,
    suite: str | None = None,
    figure_dir: str | Path = FIGURE_DIR,
    artifact_root: str | Path | None = "reports/artifacts",
    include_artifacts: bool = True,
    corrected_artifact_root: str | Path | None = None,
    math_label_mode: str = "original",
) -> list[Path]:
    _configure_style()
    runs = load_paper_runs(
        suite=suite,
        run_root=Path(artifact_root) / "__no_reports_runs__" if artifact_root is not None else "reports/runs",
        artifact_root=artifact_root,
        include_artifacts=include_artifacts,
        corrected_artifact_root=corrected_artifact_root,
        math_label_mode=math_label_mode,
    )
    runs = [run for run in runs if run.model != "mock-model"]
    include_suite = suite is None and len({run.suite for run in runs if run.suite}) > 1
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(_figure1_pipeline(figure_dir))
    outputs.extend(_figure2_success(runs, suite, figure_dir))
    outputs.extend(_figure3_calibration(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure4_budget_error_distribution(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure4_scatter(runs, figure_dir, include_suite=include_suite, output_name="appendix_tokencapbench_scatter"))
    outputs.extend(_figure5_baselines(figure_dir))
    outputs.extend(_figure6_regret(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure6_normalized_regret(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure7_cost_coverage(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure8_diagnostics(runs, figure_dir, include_suite=include_suite))
    outputs.extend(_figure_repeatability_audit(figure_dir))
    outputs.extend(_figure_fresh_coding(figure_dir))
    outputs.extend(_figure_fresh_coding_200(figure_dir))
    outputs.extend(_figure_fresh_coding_300(figure_dir))
    outputs.extend(_figure_allocation_frontier(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_fixed_budget_scheduling(figure_dir))
    outputs.extend(_figure_token_usage_proxy(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_token_usage_proxy_300(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_learned_calibration_baseline(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_forecast_stability(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_bigcodebench_hard(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_canitedit(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_replacement_token_usage_proxy(figure_dir, artifact_root=artifact_root))
    outputs.extend(_figure_replacement_allocation(figure_dir))
    outputs.extend(_figure_agentic_bridge(figure_dir))
    outputs.extend(_figure_swe_official_mini(figure_dir))
    return outputs


def _figure1_pipeline(figure_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.4, 2.35))
    ax.axis("off")
    labels = ["Task", "Forecast\ncurve", "Fresh solver runs\nunder hard caps", "Verifier", "Calibration\nand regret"]
    colors = ["#f8fafc", "#eff6ff", "#ecfdf5", "#fff7ed", "#f5f3ff"]
    edges = ["#64748b", "#2563eb", "#059669", "#ea580c", "#7c3aed"]
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.38,rounding_size=0.12",
                "facecolor": colors[index],
                "edgecolor": edges[index],
                "linewidth": 1.35,
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.085, 0.5),
                xytext=(x + 0.085, 0.5),
                arrowprops={"arrowstyle": "->", "linewidth": 1.3, "color": "#374151"},
            )
    ax.set_title("TokenCapBench protocol", fontsize=13.5, fontweight="bold", pad=12)
    return _save(fig, figure_dir / "paper_figure1_pipeline")


def _figure2_success(runs, suite: str | None, figure_dir: Path) -> list[Path]:
    bootstrap_rows = _read_csv(Path("reports/tables/bootstrap_success_by_budget.csv"))
    if bootstrap_rows:
        rows = [row for row in bootstrap_rows if not suite or row.get("suite") == suite]
        used_ci = True
    else:
        rows = _success_rows_from_runs(runs, suite)
        used_ci = False
    _write_plot_qa_note(
        figure_dir,
        "Figure 2 CI data: "
        + (
            "used reports/tables/bootstrap_success_by_budget.csv for 95% confidence bands."
            if used_ci
            else "bootstrap_success_by_budget.csv missing; fell back to point success curves without CI bands."
        ),
    )
    sources = [source for source in sorted({row.get("source", "all") for row in rows}) if source != "all"] or ["all"]
    ncols = min(3, len(sources))
    nrows = int(np.ceil(len(sources) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 3.35 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, source in zip(axes.ravel(), sources):
        ax.axis("on")
        for model in sorted({row.get("model", "") for row in rows if row.get("source") == source}):
            model_rows = sorted(
                [row for row in rows if row.get("source") == source and row.get("model") == model],
                key=lambda row: int(float(row["budget"])),
            )
            if not model_rows:
                continue
            budgets = [int(float(row["budget"])) for row in model_rows]
            values = [float(row.get("success_rate") or row.get("bootstrap_mean") or 0.0) for row in model_rows]
            ax.plot(
                budgets,
                values,
                marker=MODEL_MARKERS.get(model, "o"),
                linewidth=2.0,
                markersize=4.8,
                color=MODEL_COLORS.get(model, "#4b5563"),
                label=_short_model(model),
            )
            if model_rows[0].get("ci_low") not in {None, ""}:
                lows = [float(row["ci_low"]) for row in model_rows]
                highs = [float(row["ci_high"]) for row in model_rows]
                ax.fill_between(budgets, lows, highs, alpha=0.14, color=MODEL_COLORS.get(model, "#4b5563"))
        ax.set_title(_short_source(source), fontweight="bold")
        ax.set_xlabel("Generated-token budget")
        ax.set_ylabel("Verified success rate")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.75)
        ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    fig.suptitle("Verified success rises with token budget, but curves differ by model and suite.", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "paper_figure2_success_by_budget")


def _figure3_calibration(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for run in runs:
        for row in run.outcomes:
            meta = row.get("metadata") or {}
            groups[(run.suite or "", str(meta.get("source") or "unknown"))].append(run)
            break
    if not groups:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.text(0.5, 0.5, "No calibration points", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _save(fig, figure_dir / "paper_figure3_calibration_by_suite")
    keys = sorted(groups)
    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.45 * ncols, 3.65 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, key in zip(axes.ravel(), keys):
        ax.axis("on")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#6b7280")
        unique_runs = {str(run.run_dir): run for run in groups[key]}
        for run in sorted(unique_runs.values(), key=lambda item: item.model):
            curves = forecast_curves(run.forecasts)
            outcomes = outcomes_by_task(run.outcomes)
            probs, labels = probability_label_pairs(curves, outcomes)
            points = calibration_points(probs, labels, n_bins=8)
            if not points:
                continue
            xs, ys, counts = zip(*points)
            ax.plot(
                xs,
                ys,
                marker=MODEL_MARKERS.get(run.model, "o"),
                linewidth=1.8,
                markersize=4.4,
                color=MODEL_COLORS.get(run.model, "#4b5563"),
                label=_short_model(run.model),
            )
            for x, y, count in points:
                if count >= 300:
                    ax.annotate(str(count), (x, y), fontsize=6.5, xytext=(3, 2), textcoords="offset points", color="#4b5563")
        suite_name, source = key
        ax.set_title(f"{_short_suite(suite_name)} / {_short_source(source)}", fontweight="bold")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed success")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.75)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Calibration by Suite and Source", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, figure_dir / "paper_figure3_calibration_by_suite")
    legacy = []
    for path in paths:
        target = figure_dir / f"paper_figure3_calibration{path.suffix}"
        shutil.copyfile(path, target)
        legacy.append(target)
    return paths + legacy


def _figure4_budget_error_distribution(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: dict[tuple[str, str], str] = {}
    for run in runs:
        medians = forecast_medians(run.forecasts)
        outcomes = outcomes_by_task(run.outcomes)
        for task_id, predicted in medians.items():
            if predicted is None or predicted <= 0 or task_id not in outcomes:
                continue
            observed = observed_first_success_budget(outcomes[task_id])
            if observed is None or observed <= 0:
                continue
            key = (run.suite or "", run.model)
            grouped[key].append(float(np.log(predicted) - np.log(observed)))
            labels[key] = _run_label(run, include_suite=include_suite)
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    if not grouped:
        ax.text(0.5, 0.5, "No solved first-success-budget errors", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        keys = sorted(grouped, key=lambda key: (key[0], labels[key]))
        data = [grouped[key] for key in keys]
        positions = np.arange(len(keys))
        parts = ax.violinplot(data, positions=positions, vert=False, widths=0.78, showmedians=True, showextrema=False)
        for body, key in zip(parts["bodies"], keys):
            body.set_facecolor(MODEL_COLORS.get(key[1], "#4b5563"))
            body.set_edgecolor("white")
            body.set_alpha(0.72)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#111827")
            parts["cmedians"].set_linewidth(1.6)
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.text(0.02, 0.98, "underbudget", transform=ax.transAxes, ha="left", va="top", color="#b45309", fontsize=8)
        ax.text(0.98, 0.98, "overbudget", transform=ax.transAxes, ha="right", va="top", color="#2563eb", fontsize=8)
        ax.set_yticks(positions, [labels[key] for key in keys], fontsize=7)
        ax.set_xlabel("Signed log budget error: log(predicted) - log(observed)")
        ax.set_title("Budget Error Distribution", fontweight="bold")
        ax.grid(axis="x", alpha=0.75)
    fig.suptitle("Signed token-budget errors reveal underbudgeting and overbudgeting", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, figure_dir / "paper_figure4_budget_error_distribution")
    return paths


def _figure4_scatter(runs, figure_dir: Path, *, include_suite: bool = False, output_name: str = "paper_figure4_tokencapbench_scatter") -> list[Path]:
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for run in runs:
        source = "unknown"
        for row in run.outcomes:
            source = str((row.get("metadata") or {}).get("source") or "unknown")
            break
        groups[(run.suite or "", source)].append(run)
    if not groups:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.text(0.5, 0.5, "No success-budget points", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _save(fig, figure_dir / output_name)
    keys = sorted(groups)
    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.45 * ncols, 3.75 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, key in zip(axes.ravel(), keys):
        ax.axis("on")
        plotted = False
        all_values: list[float] = []
        for run in groups[key]:
            medians = forecast_medians(run.forecasts)
            outcomes = outcomes_by_task(run.outcomes)
            solved_x, solved_y, cens_x, cens_y = [], [], [], []
            for task_id, predicted in medians.items():
                if predicted is None or predicted <= 0 or task_id not in outcomes:
                    continue
                observed = observed_first_success_budget(outcomes[task_id])
                if observed is not None:
                    solved_x.append(predicted)
                    solved_y.append(observed)
                    all_values.extend([float(predicted), float(observed)])
                else:
                    censored_at = observed_censored_at_budget(outcomes[task_id])
                    if censored_at is not None:
                        cens_x.append(predicted)
                        cens_y.append(censored_at)
                        all_values.extend([float(predicted), float(censored_at)])
            if solved_x:
                ax.scatter(solved_x, solved_y, alpha=0.42, s=20, color="#2563eb", edgecolors="none")
                plotted = True
            if cens_x:
                ax.scatter(cens_x, cens_y, alpha=0.72, s=30, marker="^", color="#b45309", edgecolors="white", linewidths=0.25)
                plotted = True
        if plotted and all_values:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo = max(1.0, min(all_values) * 0.75)
            hi = max(all_values) * 1.25
            ax.plot([lo, hi], [lo, hi], color="#6b7280", linestyle="--", linewidth=1.0)
            ax.scatter([], [], alpha=0.55, s=18, color="#2563eb", label="solved")
            ax.scatter([], [], alpha=0.55, s=26, marker="^", color="#b45309", label="censored")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        else:
            ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
        suite_name, source = key
        ax.set_title(f"{_short_suite(suite_name)} / {_short_source(source)}", fontweight="bold")
        ax.set_xlabel("Predicted success budget")
        ax.set_ylabel("Observed or censoring budget")
        ax.grid(True, alpha=0.75)
    fig.suptitle("Predicted vs Observed TokenCapBench", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, figure_dir / output_name)
    if output_name != "paper_figure4_tokencapbench_scatter":
        legacy = []
        for path in paths:
            target = figure_dir / f"paper_figure4_tokencapbench_scatter{path.suffix}"
            shutil.copyfile(path, target)
            legacy.append(target)
        paths.extend(legacy)
    return paths


def _figure5_baselines(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/baseline_comparison.csv"))
    deployable = _figure5_baseline_class(rows, "calibration", figure_dir / "paper_figure5_calibration_split_baselines")
    diagnostic = _figure5_baseline_class(rows, "diagnostic", figure_dir / "appendix_diagnostic_baselines")
    legacy = []
    for path in deployable:
        target = figure_dir / f"paper_figure5_baselines{path.suffix}"
        shutil.copyfile(path, target)
        legacy.append(target)
        target2 = figure_dir / f"paper_figure5_deployable_baselines{path.suffix}"
        shutil.copyfile(path, target2)
        legacy.append(target2)
    for path in diagnostic:
        target = figure_dir / f"paper_figure5b_diagnostic_baselines{path.suffix}"
        shutil.copyfile(path, target)
        legacy.append(target)
    return deployable + diagnostic + legacy


def _figure5_baseline_class(rows: list[dict[str, str]], baseline_class: str, output_prefix: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    if baseline_class == "calibration":
        accepted = {"model_forecast_raw", "model_forecast_recalibrated", "calibration_split_baseline"}
    else:
        accepted = {"test_distribution_diagnostic", "posthoc_diagnostic"}
    class_rows = [
        row
        for row in rows
        if (row.get("baseline_class") or ("posthoc_diagnostic" if row.get("forecast_method") == "output_length_proxy" else "model_forecast_raw"))
        in accepted
        and row.get("track") == "all"
        and row.get("source") == "all"
    ]
    if not class_rows:
        ax.text(0.5, 0.5, f"No {baseline_class} baseline rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        if baseline_class == "calibration":
            order = [
                "self_forecast_raw",
                "self_forecast_histogram_recalibrated",
                "constant_by_budget_calibration",
                "source_by_budget_calibration",
                "prompt_length_bin_calibration",
                "single_budget_midpoint",
            ]
        else:
            order = ["test_distribution_leave_one_out_source", "output_length_proxy_posthoc", "output_length_proxy"]
        methods = [method for method in order if any(row["forecast_method"] == method for row in class_rows)]
        methods.extend(sorted({row["forecast_method"] for row in class_rows} - set(methods)))
        values: list[float] = []
        error_lows: list[float] = []
        error_highs: list[float] = []
        has_errors = False
        for method in methods:
            method_rows = [
                row for row in class_rows if row["forecast_method"] == method and row.get("brier") not in {"", None}
            ]
            method_values = [float(row["brier"]) for row in method_rows]
            value = float(np.nanmean(method_values)) if method_values else float("nan")
            lows = [
                float(row["brier_ci_low"])
                for row in method_rows
                if row.get("brier_ci_low") not in {"", None}
            ]
            highs = [
                float(row["brier_ci_high"])
                for row in method_rows
                if row.get("brier_ci_high") not in {"", None}
            ]
            low = float(np.nanmean(lows)) if lows else value
            high = float(np.nanmean(highs)) if highs else value
            values.append(value)
            error_lows.append(max(0.0, value - low) if np.isfinite(value) and np.isfinite(low) else 0.0)
            error_highs.append(max(0.0, high - value) if np.isfinite(value) and np.isfinite(high) else 0.0)
            has_errors = has_errors or bool(lows and highs)
        y = np.arange(len(methods))
        color = "#2563eb" if baseline_class == "calibration" else "#b45309"
        if has_errors:
            ax.barh(
                y,
                values,
                xerr=np.asarray([error_lows, error_highs]),
                color=color,
                alpha=0.9,
                error_kw={"ecolor": "#111827", "elinewidth": 1.0, "capsize": 3},
            )
        else:
            ax.barh(y, values, color=color, alpha=0.9)
        ax.set_yticks(y, [_pretty_method(method) for method in methods])
        ax.invert_yaxis()
        finite_values = [value for value in values if np.isfinite(value)]
        label_offset = (max(finite_values) * 0.015) if finite_values else 0.01
        for y_pos, value in zip(y, values):
            if np.isfinite(value):
                ax.text(value + label_offset, y_pos, f"{value:.3f}", va="center", fontsize=8, color="#374151")
        ax.set_title("Calibration-Split Baselines" if baseline_class == "calibration" else "Diagnostic Baselines", fontweight="bold")
        ax.set_xlabel("Mean Brier score")
        ax.set_ylabel("Mean Brier score")
        ax.set_ylabel("")
        finite_highs = [value + high for value, high in zip(values, error_highs) if np.isfinite(value)]
        limit_basis = finite_highs or finite_values
        ax.set_xlim(0, max(limit_basis) * 1.18 if limit_basis else 1)
        ax.grid(axis="x", alpha=0.75)
    fig.tight_layout()
    return _save(fig, output_prefix)


def _figure6_regret(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    suites = sorted({run.suite or "" for run in runs})
    ncols = min(3, len(suites) or 1)
    nrows = int(np.ceil((len(suites) or 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 3.45 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    for ax, suite_name in zip(axes.ravel(), suites or [""]):
        ax.axis("on")
        for run in [item for item in runs if (item.suite or "") == suite_name]:
            curves = forecast_curves(run.forecasts)
            outcomes = outcomes_by_task(run.outcomes)
            y_values = []
            for lam in lambdas:
                regrets = []
                for task_id, curve in curves.items():
                    task_outcomes = outcomes.get(task_id)
                    if not task_outcomes:
                        continue
                    chosen = selected_budget_from_forecast(curve, reward=1.0, token_cost=lam)
                    regrets.append(
                        oracle_utility(task_outcomes, reward=1.0, token_cost=lam)
                        - utility(task_outcomes.get(chosen, False), chosen, reward=1.0, token_cost=lam)
                    )
                y_values.append(float(np.mean(regrets)) if regrets else 0.0)
            ax.plot(
                lambdas,
                y_values,
                marker=MODEL_MARKERS.get(run.model, "o"),
                linewidth=1.9,
                markersize=4.4,
                color=MODEL_COLORS.get(run.model, "#4b5563"),
                label=_short_model(run.model),
            )
        ax.set_title(_short_suite(suite_name), fontweight="bold")
        ax.set_xlabel("Token cost lambda")
        ax.set_ylabel("Mean regret")
        ax.set_xscale("symlog", linthresh=1e-7)
        ax.grid(True, alpha=0.75)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Budget-Selection Regret by Suite", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "paper_figure6_regret")


def _figure6_normalized_regret(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        rows.append({"label": _run_label(run, include_suite=include_suite), "value": scored.get("normalized_regret")})
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    rows = [row for row in rows if row.get("value") not in {None, ""}]
    if not rows:
        ax.text(0.5, 0.5, "No normalized regret values", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        rows = sorted(rows, key=lambda row: float(row["value"]))
        y = np.arange(len(rows))
        ax.barh(y, [float(row["value"]) for row in rows], color="#0f766e", alpha=0.9)
        ax.set_yticks(y, [row["label"] for row in rows], fontsize=7)
        ax.set_xlabel("Normalized regret")
        ax.set_title("Normalized Budget-Selection Regret", fontweight="bold")
        ax.grid(axis="x", alpha=0.75)
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_normalized_regret")


def _figure7_cost_coverage(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    costs = _live_costs()
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    if not runs:
        ax.text(0.5, 0.5, "No run artifacts", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    for index, run in enumerate(runs):
        cost = costs.get(run.run_id, _cost_from_usage(run.model, run.forecasts, run.outcomes))
        outcome_count = len(run.outcomes)
        label = _run_label(run, include_suite=include_suite)
        ax.scatter(
            outcome_count,
            cost,
            s=78,
            color=MODEL_COLORS.get(run.model, "#4b5563"),
            marker=MODEL_MARKERS.get(run.model, "o"),
            alpha=0.88,
            edgecolors="white",
            linewidths=0.6,
            label=_short_model(run.model),
        )
    ax.set_title("Cost and Coverage", fontweight="bold")
    ax.set_xlabel("Budgeted solver outcomes")
    ax.set_ylabel("Estimated API cost (USD)")
    ax.text(
        0.01,
        0.98,
        "Costs use logged token usage and pricing; ledger CSV overrides estimates when present.",
        transform=ax.transAxes,
        fontsize=7,
        color="#374151",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    ax.grid(True, alpha=0.75)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return _save(fig, figure_dir / "paper_figure7_cost_coverage")


def _figure8_diagnostics(runs, figure_dir: Path, *, include_suite: bool = False) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        rows.append(
            {
                "label": _run_label(run, include_suite=include_suite),
                "forecast_monotonicity": scored.get("forecast_monotonicity_violation_rate") or 0.0,
                "outcome_nonmonotonicity": scored.get("outcome_nonmonotonicity_rate") or 0.0,
                "truncation": scored.get("truncation_rate") or 0.0,
                "ranking_accuracy": scored.get("task_budget_ranking_accuracy") or 0.0,
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4))
    if not rows:
        axes[0].text(0.5, 0.5, "No diagnostics", ha="center", va="center", transform=axes[0].transAxes)
        for ax in axes:
            ax.axis("off")
    else:
        rows = sorted(rows, key=lambda row: row["truncation"])
        labels = [row["label"] for row in rows]
        y = np.arange(len(labels))
        height = 0.23
        axes[0].barh(y - height, [row["forecast_monotonicity"] for row in rows], height, label="Forecast decreases", color="#2563eb")
        axes[0].barh(y, [row["outcome_nonmonotonicity"] for row in rows], height, label="Outcome reversals", color="#7c3aed")
        axes[0].barh(y + height, [row["truncation"] for row in rows], height, label="Truncated outputs", color="#059669")
        axes[0].set_title("Artifact Diagnostics", fontweight="bold")
        axes[0].set_xlabel("Rate")
        axes[0].set_xlim(0, 1)
        axes[0].set_yticks(y, labels, fontsize=6.8)
        axes[0].grid(axis="x", alpha=0.75)
        axes[0].legend(frameon=False, fontsize=7)
        rows_rank = sorted(rows, key=lambda row: row["ranking_accuracy"])
        labels_rank = [row["label"] for row in rows_rank]
        axes[1].barh(labels_rank, [row["ranking_accuracy"] for row in rows_rank], color="#0f766e")
        axes[1].set_title("Task-Budget Ranking Accuracy", fontweight="bold")
        axes[1].set_xlabel("Pairwise accuracy")
        axes[1].set_xlim(0, 1)
        axes[1].tick_params(axis="y", labelsize=6.8)
        axes[1].grid(axis="x", alpha=0.75)
    fig.tight_layout()
    return _save(fig, figure_dir / "paper_figure8_diagnostics")


def _figure_repeatability_audit(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table10_repeatability.csv")) or _read_csv(Path("reports/tables/paper_table_repeatability_audit.csv"))
    if not rows:
        fig, ax = plt.subplots(figsize=(8.8, 4.6))
        ax.text(0.5, 0.5, "Repeatability audit not run", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        rows = sorted(rows, key=lambda row: (row.get("source", ""), int(float(row.get("budget") or 0)), row.get("model", "")))
        fig_height = max(4.8, 1.2 + 0.32 * len(rows))
        fig, ax = plt.subplots(figsize=(9.6, fig_height))
        labels = [
            f"{_short_source(row.get('source', ''))} {int(float(row.get('budget') or 0))} - {_short_model(row.get('model', ''))}"
            for row in rows
        ]
        values = [float(row.get("success_agreement_rate") or 0.0) for row in rows]
        y = np.arange(len(rows))
        ax.barh(y, values, color="#0f766e", alpha=0.9)
        ax.set_yticks(y, labels, fontsize=7.0)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Success agreement across repeats")
        ax.set_title("Repeatability Audit", fontweight="bold")
        ax.grid(axis="x", alpha=0.75)
    fig.tight_layout()
    paths = _save(fig, figure_dir / "appendix_repeatability_variance")
    legacy = []
    for path in paths:
        target = figure_dir / f"appendix_repeatability_audit{path.suffix}"
        shutil.copyfile(path, target)
        legacy.append(target)
    return paths + legacy


def _figure_fresh_coding(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table11_fresh_coding.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    completed = [row for row in rows if row.get("official_harness_status") == "completed"]
    if completed:
        labels = [_short_model(row.get("model", "")) for row in completed]
        values = [float(row.get("success_at_max_budget") or 0.0) for row in completed]
        ax.barh(np.arange(len(labels)), values, color="#2563eb", alpha=0.9)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Success at max budget")
        ax.set_title("Fresh Coding Official Run", fontweight="bold")
        ax.grid(axis="x", alpha=0.75)
    else:
        n_tasks = rows[0].get("n_tasks") if rows else "0"
        ax.text(
            0.5,
            0.55,
            f"LiveCodeBench fresh split configured\nOfficial run not included\nTasks: {n_tasks}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_fresh_coding")


def _figure_fresh_coding_200(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table11b_fresh_coding_200.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    completed = [
        row for row in rows
        if row.get("run_status", "").startswith("completed") and row.get("success_at_max_budget") not in {"", None}
    ]
    if completed:
        labels = [_short_model(row.get("model") or row.get("models") or "") for row in completed]
        values = [float(row.get("success_at_max_budget") or 0.0) for row in completed]
        ax.barh(np.arange(len(labels)), values, color="#2563eb")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Success at max budget")
    else:
        local_tasks = rows[0].get("local_tasks_available", "0") if rows else "0"
        cost = rows[0].get("estimated_cost_usd", "NA") if rows else "NA"
        status = rows[0].get("run_status", "not_run") if rows else "not_run"
        ax.text(
            0.5,
            0.55,
            f"Expanded LiveCodeBench status: {status}\nLocal tasks: {local_tasks}; estimated cost: ${cost}\nOfficial labels required for paper claims",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    ax.set_title("Expanded Fresh Coding Status", fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_fresh_coding_200")


def _figure_fresh_coding_300(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table11c_fresh_coding_300.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    completed = [
        row for row in rows
        if row.get("run_status", "").startswith("completed") and row.get("success_at_max_budget") not in {"", None}
    ]
    if completed:
        labels = [_short_model(row.get("model") or row.get("models") or "") for row in completed]
        values = [float(row.get("success_at_max_budget") or 0.0) for row in completed]
        order = np.argsort(values)
        ax.barh(np.arange(len(labels)), [values[idx] for idx in order], color="#2563eb", alpha=0.9)
        ax.set_yticks(np.arange(len(labels)), [labels[idx] for idx in order])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Success at max budget")
    else:
        local_tasks = rows[0].get("local_tasks_available", "0") if rows else "0"
        cost = rows[0].get("estimated_cost_usd", "NA") if rows else "NA"
        status = rows[0].get("run_status", "not_run") if rows else "not_run"
        ax.text(
            0.5,
            0.55,
            f"LiveCodeBench-300 status: {status}\nLocal tasks: {local_tasks}; estimated cost: ${cost}\nOfficial labels required for paper claims",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    ax.set_title("LiveCodeBench-300 Fresh Coding", fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_fresh_coding_300")


def _figure_agentic_bridge(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/appendix_agentic_bridge.csv"))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    if rows and rows[0].get("run_status", "").startswith("completed"):
        budgets = sorted(
            int(key.replace("success_at_", ""))
            for key, value in rows[0].items()
            if key.startswith("success_at_") and value not in {"", None}
        ) or [4096, 16384]
        successes = [float(rows[0].get(f"success_at_{budget}") or 0.0) for budget in budgets]
        ax.plot(budgets, successes, marker="o", linewidth=2.0, color="#2563eb")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Generated-token cap")
        ax.set_ylabel("Success rate")
    else:
        cost = rows[0].get("estimated_cost_usd", "NA") if rows else "NA"
        ax.text(
            0.5,
            0.55,
            f"Agentic SWE bridge not executed\nEstimated cost: ${cost}\nGenerated-token cap pilot only",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    ax.set_title("Agentic Bridge Status", fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_agentic_bridge_success_by_budget")


def _figure_allocation_frontier(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    table_path = Path("reports/tables/paper_table12_allocation_frontier.csv")
    if not table_path.exists() or not table_path.read_text(encoding="utf-8").strip():
        if artifact_root is None:
            return []
        run_allocation_frontier(
            artifact_root=artifact_root,
            split_dir="reports/splits",
            output_table=table_path,
            figures_dir=figure_dir,
            write_figures=False,
        )
    return plot_allocation_frontier(table_path=table_path, figures_dir=figure_dir)


def _figure_fixed_budget_scheduling(figure_dir: Path) -> list[Path]:
    table_path = Path("reports/tables/paper_table15_fixed_budget_scheduling.csv")
    if not table_path.exists() or not table_path.read_text(encoding="utf-8").strip():
        return []
    return plot_fixed_budget_scheduling(table_path=table_path, figures_dir=figure_dir)


def _figure_token_usage_proxy(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    if artifact_root is None:
        return []
    outputs = analyze_token_usage_proxy(
        artifact_root=artifact_root,
        split_dir="reports/splits",
        output_table="reports/tables/paper_table13_token_usage_proxy.csv",
        output_figure_prefix=figure_dir / "paper_figure10_token_usage_proxy_vs_success",
        write_figure=True,
    )
    return [path for path in outputs if Path(path).suffix in {".png", ".svg"}]


def _figure_token_usage_proxy_300(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    if artifact_root is None:
        return []
    outputs = analyze_token_usage_proxy(
        artifact_root=artifact_root,
        split_dir="reports/splits",
        dual_forecast_root="reports/runs/paper_dual_success_usage_forecast_300",
        output_table="reports/tables/paper_table13b_token_usage_proxy_300.csv",
        output_figure_prefix=figure_dir / "paper_figure10b_token_usage_proxy_300",
        write_figure=True,
    )
    return [path for path in outputs if Path(path).suffix in {".png", ".svg"}]


def _figure_bigcodebench_hard(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table18_bigcodebench_hard.csv"))
    paths = _plot_replacement_success_runs(
        suite="paper_bigcodebench_hard",
        fallback_rows=rows,
        artifact_root=artifact_root,
        output_prefix=figure_dir / "paper_figure11_bigcodebench_hard_success_by_budget",
        title="BigCodeBench-Hard Success by Budget",
        status_key="official_harness_status",
    )
    if any(row.get("official_harness_status") == "official_labels_completed" for row in rows):
        paths.extend(_copy_figure_aliases(paths, figure_dir / "paper_figure14_bigcodebench_success_by_budget"))
    return paths


def _figure_canitedit(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    rows = _read_csv(Path("reports/tables/paper_table19_canitedit_descriptive.csv"))
    paths = _plot_replacement_success_runs(
        suite="paper_canitedit_descriptive",
        fallback_rows=rows,
        artifact_root=artifact_root,
        output_prefix=figure_dir / "paper_figure12_canitedit_descriptive_success_by_budget",
        title="CanItEdit Descriptive Success by Budget",
        status_key="verifier_status",
    )
    paths.extend(_copy_figure_aliases(paths, figure_dir / "paper_figure13_canitedit_success_by_budget"))
    return paths


def _figure_replacement_token_usage_proxy(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    if artifact_root is None:
        return []
    outputs = analyze_token_usage_proxy(
        artifact_root=artifact_root,
        split_dir="reports/splits",
        output_table="reports/tables/paper_table20_replacement_token_usage_proxy.csv",
        output_figure_prefix=figure_dir / "paper_figure13_replacement_token_proxy_vs_success",
        write_figure=True,
        suite_filter={"paper_bigcodebench_hard", "paper_canitedit_descriptive"},
    )
    return [path for path in outputs if Path(path).suffix in {".png", ".svg"}]


def _figure_replacement_allocation(figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    raw_table = Path("reports/tables/paper_table21_replacement_allocation_frontier_raw.csv")
    if raw_table.exists() and raw_table.read_text(encoding="utf-8").strip():
        raw_paths = _plot_replacement_frontier_raw(
            table_path=raw_table,
            output_prefix=figure_dir / "paper_figure11_replacement_allocation_frontier",
        )
    else:
        raw_paths = _save_empty(figure_dir / "paper_figure11_replacement_allocation_frontier", "No replacement allocation frontier rows")
    paths.extend(raw_paths)
    paths.extend(_copy_figure_aliases(raw_paths, figure_dir / "paper_figure14_replacement_allocation_frontier"))

    table_path = Path("reports/tables/paper_table21_replacement_fixed_budget_scheduling.csv")
    if not table_path.exists() or not table_path.read_text(encoding="utf-8").strip():
        fixed_paths = _save_empty(figure_dir / "paper_figure12_replacement_fixed_budget_scheduling", "No replacement fixed-budget rows")
    else:
        fixed_paths = plot_replacement_allocation_frontier(
            table_path=table_path,
            figures_dir=figure_dir,
            output_prefix=figure_dir / "paper_figure12_replacement_fixed_budget_scheduling",
        )
    paths.extend(fixed_paths)
    return paths


def _plot_replacement_frontier_raw(*, table_path: Path, output_prefix: Path) -> list[Path]:
    rows = _read_csv(table_path)
    suites = sorted({row.get("suite", "") for row in rows if row.get("suite")})
    if not suites:
        return _save_empty(output_prefix, "No replacement allocation frontier rows")
    methods = [
        "oracle",
        "self_forecast_raw",
        "self_forecast_histogram_recalibrated",
        "source_by_budget_calibration",
        "random_budget",
    ]
    ncols = min(2, len(suites))
    nrows = int(np.ceil(len(suites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.1 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, suite in zip(axes.ravel(), suites):
        ax.axis("on")
        suite_rows = [row for row in rows if row.get("suite") == suite]
        max_budget = max((_safe_float(row.get("total_budget")) or 0.0) for row in suite_rows) or 1.0
        for method in methods:
            method_rows = [row for row in suite_rows if row.get("method") == method and row.get("policy") == "policy_a"]
            if not method_rows:
                continue
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in method_rows:
                total_budget = _safe_float(row.get("total_budget"))
                success_rate = _safe_float(row.get("success_rate"))
                if total_budget is None or success_rate is None:
                    continue
                grouped[round(total_budget / max_budget, 3)].append(success_rate)
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            ax.plot(
                [x * 100.0 for x in xs],
                ys,
                marker=_method_marker(method),
                linewidth=2.2,
                markersize=4.6,
                color=_method_color(method),
                label=_frontier_method_label(method),
            )
        ax.set_title(_suite_label(suite), fontweight="bold")
        ax.set_xlabel("Global token budget used (%)")
        ax.set_ylabel("Mean verified success rate")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.35)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Replacement-track allocation frontier", fontsize=13.2, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return _save(fig, output_prefix)


def _plot_replacement_success_runs(
    *,
    suite: str,
    fallback_rows: list[dict[str, str]],
    artifact_root: str | Path | None,
    output_prefix: Path,
    title: str,
    status_key: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    try:
        roots = [Path(artifact_root or "reports/artifacts")]
        if suite == "paper_bigcodebench_hard":
            official_root = Path("reports/artifacts_bigcodebench_official/paper_bigcodebench_hard")
            if official_root.exists():
                roots.append(official_root)
        runs = []
        for root in roots:
            runs.extend(
                load_paper_runs(
                    suite=suite,
                    run_root="reports/runs",
                    artifact_root=root,
                    include_artifacts=True,
                )
            )
    except FileNotFoundError:
        runs = []
    runs = [run for run in runs if run.model != "mock-model"]
    deduped_runs = {}
    for run in runs:
        deduped_runs[run.model] = run
    runs = list(deduped_runs.values())
    plotted = False
    for run in runs:
        if not run.outcomes:
            continue
        if suite == "paper_bigcodebench_hard" and not all(
            (row.get("metadata") or {}).get("label_source") == "official_bigcodebench" for row in run.outcomes
        ):
            continue
        grouped: dict[int, list[bool]] = defaultdict(list)
        for row in run.outcomes:
            if row.get("budget") is not None:
                grouped[int(row["budget"])].append(bool(row.get("success")))
        if not grouped:
            continue
        budgets = sorted(grouped)
        values = [sum(grouped[budget]) / len(grouped[budget]) for budget in budgets]
        label = f"{_short_model(run.model)} ({values[-1]:.0%})"
        ax.plot(
            budgets,
            values,
            marker=MODEL_MARKERS.get(run.model, "o"),
            linewidth=2.35,
            markersize=5.2,
            label=label,
            color=MODEL_COLORS.get(run.model, "#4b5563"),
        )
        plotted = True
    if plotted:
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Generated-token cap")
        ax.set_ylabel("Verified success rate")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    else:
        status = fallback_rows[0].get(status_key, "not_run") if fallback_rows else "not_run"
        n_tasks = fallback_rows[0].get("n_tasks", "0") if fallback_rows else "0"
        ax.text(
            0.5,
            0.55,
            f"{title}\nStatus: {status}\nTasks available: {n_tasks}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_prefix)


def _figure_learned_calibration_baseline(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    table_path = Path("reports/tables/paper_table16_learned_calibration_baseline.csv")
    if not table_path.exists() or not table_path.read_text(encoding="utf-8").strip():
        if artifact_root is None:
            return []
        outputs = run_learned_calibration_baseline(
            artifact_root=artifact_root,
            split_dir="reports/splits",
            output_table=table_path,
            output_figure_prefix=figure_dir / "paper_figure12_learned_calibration_baseline",
            n_bootstrap=250,
            write_figure=True,
        )
        return [path for path in outputs if Path(path).suffix in {".png", ".svg"}]
    return plot_learned_calibration_baseline(
        table_path=table_path,
        output_figure_prefix=figure_dir / "paper_figure12_learned_calibration_baseline",
    )


def _figure_forecast_stability(figure_dir: Path, *, artifact_root: str | Path | None) -> list[Path]:
    if artifact_root is None:
        return []
    outputs = analyze_forecast_stability(
        artifact_root=artifact_root,
        output_table="reports/tables/paper_table14_forecast_stability.csv",
        output_figure_prefix=figure_dir / "appendix_forecast_stability",
    )
    return [path for path in outputs if Path(path).suffix in {".png", ".svg"}]


def _figure_swe_official_mini(figure_dir: Path) -> list[Path]:
    rows = _read_csv(Path("reports/tables/appendix_swe_official_mini.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    completed = [row for row in rows if row.get("official_harness_status") in {"completed", "failed_or_incomplete"}]
    if completed:
        budget_keys = sorted(
            {
                int(key.replace("success_at_", ""))
                for row in completed
                for key, value in row.items()
                if key.startswith("success_at_") and key.replace("success_at_", "").isdigit() and value not in {"", None}
            }
        )
        for row in completed:
            values = [float(row.get(f"success_at_{budget}") or 0.0) for budget in budget_keys]
            if not budget_keys:
                continue
            ax.plot(
                budget_keys,
                values,
                marker=MODEL_MARKERS.get(row.get("model"), "o"),
                linewidth=2.0,
                label=_short_model(row.get("model", "")),
                color=MODEL_COLORS.get(row.get("model"), "#4b5563"),
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Generated-token cap")
        ax.set_ylabel("Official SWE-bench success rate")
        ax.grid(True, alpha=0.75)
        ax.legend(frameon=False, fontsize=7.5)
        if not any(float(row.get("official_labels") or 0) > 0 for row in completed):
            ax.text(0.5, 0.08, "Official labels unavailable; all rows treated as unresolved", ha="center", transform=ax.transAxes)
    else:
        status = rows[0].get("official_harness_status", "not_run") if rows else "not_run"
        ax.text(
            0.5,
            0.55,
            f"SWE-bench Verified mini status: {status}\nOfficial harness labels required for success claims",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.axis("off")
    ax.set_title("SWE-bench Verified Mini Official Bridge", fontweight="bold")
    fig.tight_layout()
    return _save(fig, figure_dir / "appendix_swe_official_mini_success_by_budget")


def _success_rows_from_runs(runs, suite: str | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[bool]] = defaultdict(list)
    for run in runs:
        for row in run.outcomes:
            meta = row.get("metadata") or {}
            source = str(meta.get("source") or "unknown")
            track = str(meta.get("track") or "unknown")
            grouped[(run.model, track, source, int(row["budget"]))].append(bool(row["success"]))
    return [
        {
            "suite": suite or "",
            "model": model,
            "track": track,
            "source": source,
            "budget": budget,
            "success_rate": sum(values) / len(values),
        }
        for (model, track, source, budget), values in sorted(grouped.items())
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _live_costs() -> dict[str, float]:
    path = Path("reports/live_runs/provider_live_summary.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["run_id"]: float(row.get("estimated_cost_usd") or 0.0) for row in csv.DictReader(f)}


def _cost_from_usage(model: str, forecasts: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> float:
    pricing_path = Path("reports/live_runs/provider_live_cost_estimate.json")
    if not pricing_path.exists():
        return 0.0
    pricing = json.loads(pricing_path.read_text(encoding="utf-8")).get("pricing", {})
    rates = pricing.get(model)
    if not rates:
        return 0.0
    input_tokens = sum(int((row.get("metadata") or {}).get("prompt_tokens") or 0) for row in forecasts)
    output_tokens = sum(int((row.get("metadata") or {}).get("completion_tokens") or 0) for row in forecasts)
    input_tokens += sum(int(row.get("prompt_tokens") or 0) for row in outcomes)
    output_tokens += sum(int(row.get("completion_tokens") or 0) for row in outcomes)
    return (
        input_tokens / 1_000_000.0 * float(rates["input_per_m"])
        + output_tokens / 1_000_000.0 * float(rates["output_per_m"])
    )


def _short_model(model: str) -> str:
    return (
        model.replace("gemini-2.0-", "Gemini ")
        .replace("-001", "")
        .replace("DeepSeek-V3-0324", "DeepSeek V3")
        .replace("claude-3-haiku", "Claude Haiku")
        .replace("gpt-5-mini", "GPT-5 mini")
        .replace("_", " ")
    )


def _safe_float(value: Any) -> float | None:
    if value in {None, "", "NA"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frontier_method_label(method: str) -> str:
    return {
        "oracle": "Oracle",
        "self_forecast_raw": "Raw self-forecast",
        "self_forecast_histogram_recalibrated": "Recalibrated",
        "source_by_budget_calibration": "Source prior",
        "random_budget": "Random budget",
    }.get(method, method.replace("_", " "))


def _method_color(method: str) -> str:
    return {
        "oracle": "#111827",
        "self_forecast_raw": "#2563eb",
        "self_forecast_histogram_recalibrated": "#059669",
        "source_by_budget_calibration": "#7c3aed",
        "random_budget": "#6b7280",
    }.get(method, "#4b5563")


def _method_marker(method: str) -> str:
    return {
        "oracle": "o",
        "self_forecast_raw": "D",
        "self_forecast_histogram_recalibrated": "s",
        "source_by_budget_calibration": "^",
        "random_budget": "X",
    }.get(method, "o")


def _short_source(source: str) -> str:
    return {
        "evalplus_humaneval": "HumanEval+",
        "evalplus_mbpp": "MBPP+",
        "gsm8k": "GSM8K",
        "hendrycks_math": "MATH-500",
    }.get(source, source.replace("_", " "))


def _pretty_method(method: str) -> str:
    return {
        "source_empirical_prior": "Source empirical prior",
        "prompt_length_empirical": "Prompt-length empirical",
        "constant_empirical_prior": "Constant empirical prior",
        "single_budget": "Single-budget threshold",
        "output_length_proxy": "Output-length proxy",
        "self_forecast": "Self forecast",
        "self_forecast_raw": "Raw self forecast",
        "self_forecast_histogram_recalibrated": "Recalibrated self forecast",
        "constant_by_budget_calibration": "Calibration constant prior",
        "source_by_budget_calibration": "Calibration source prior",
        "prompt_length_bin_calibration": "Prompt-length bin prior",
        "single_budget_midpoint": "Single-budget midpoint",
        "test_distribution_leave_one_out_source": "Test-distribution source prior",
        "output_length_proxy_posthoc": "Output-length proxy",
    }.get(method, method.replace("_", " "))


def _run_label(run, *, include_suite: bool = False) -> str:
    label = _short_model(run.model)
    if include_suite and run.suite:
        return f"{label} {_short_suite(run.suite)}"
    return label


def _short_suite(suite: str) -> str:
    return {
        "paper_math_core": "Math",
        "paper_evalplus_humaneval_full": "HumanEval+",
        "paper_evalplus_mbpp_full": "MBPP+",
        "paper_bigcodebench_hard": "BigCodeBench-Hard",
        "paper_canitedit_descriptive": "CanItEdit Descriptive",
        "paper_livecodebench_fresh_300": "LiveCodeBench Fresh 300",
        "paper_aider_polyglot": "Aider Polyglot",
    }.get(suite, suite.replace("paper_", "").replace("_", " "))


def _suite_label(suite: str) -> str:
    return _short_suite(suite)


def _save_empty(prefix: Path, message: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, prefix)


def _save(fig, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".svg"), prefix.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return paths


def _copy_figure_aliases(paths: list[Path], alias_prefix: Path) -> list[Path]:
    aliases: list[Path] = []
    for path in paths:
        if path.suffix not in {".png", ".svg"} or not path.exists():
            continue
        target = alias_prefix.with_suffix(path.suffix)
        if path.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        aliases.append(target)
    return aliases


def _write_plot_qa_note(figure_dir: Path, note: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "paper_plot_qa_notes.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Paper Figure QA Notes\n"
    marker = "- " + note
    lines = [line for line in existing.splitlines() if "Figure 2 CI data:" not in line]
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(marker)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from frozen TokenCapBench outputs.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--corrected-artifact-root", default=None)
    parser.add_argument(
        "--math-label-mode",
        choices=["original", "strict", "corrected"],
        default="original",
        help="'strict' is retained as an alias for task-default corrected math labels.",
    )
    args = parser.parse_args()
    outputs = make_paper_figures(
        suite=args.suite,
        figure_dir=args.figure_dir,
        artifact_root=args.artifact_root,
        include_artifacts=not args.no_artifacts,
        corrected_artifact_root=args.corrected_artifact_root,
        math_label_mode=args.math_label_mode,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
