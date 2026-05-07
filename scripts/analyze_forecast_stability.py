#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import forecast_curves, infer_suite_from_artifact_dir, load_run_config, outcomes_by_task
from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.utils.jsonl import read_jsonl


TABLE_FIELDS = [
    "suite",
    "model",
    "forecast_groups",
    "tasks_with_repeats",
    "mean_probability_std",
    "selected_budget_agreement",
    "curve_rank_correlation",
    "brier_std",
    "ece_std",
    "budget_choice_change_fraction",
    "prompt_variants_observed",
    "notes",
]


def analyze_forecast_stability(
    *,
    artifact_root: str | Path = "reports/artifacts",
    live_run_root: str | Path | None = None,
    output_table: str | Path = "reports/tables/paper_table14_forecast_stability.csv",
    output_figure_prefix: str | Path = "reports/figures/appendix_forecast_stability",
) -> list[Path]:
    artifact_root_path = Path(artifact_root)
    if live_run_root is None and artifact_root_path == Path("reports/artifacts"):
        live_run_root = "reports/runs/paper_forecast_stability"
    groups = _load_repeatability_groups(artifact_root_path)
    if live_run_root is not None:
        live_groups = _load_forecast_only_groups(Path(live_run_root), _load_outcome_bank(artifact_root_path))
        for key, runs in live_groups.items():
            groups[key].extend(runs)
    rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    for (suite, model), runs in sorted(groups.items()):
        row = _score_group(suite, model, runs)
        rows.append(row)
        plot_rows.append(row)
    table_path = Path(output_table)
    _write_csv(table_path, rows)
    figure_paths = _plot_stability(plot_rows, Path(output_figure_prefix))
    return [table_path, *figure_paths]


def _load_repeatability_groups(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not root.exists():
        return groups
    for forecast_path in sorted(root.glob("paper_repeatability_small*/forecasts.jsonl")):
        run_dir = forecast_path.parent
        config = load_run_config(run_dir)
        forecasts = read_jsonl(forecast_path)
        outcomes_path = run_dir / "outcomes.jsonl"
        outcomes = read_jsonl(outcomes_path) if outcomes_path.exists() else []
        model = str(config.get("model") or _first_value(forecasts, "model") or run_dir.name)
        suite = infer_suite_from_artifact_dir(run_dir, config) or str(config.get("suite_name") or "paper_repeatability_small")
        groups[(suite, model)].append(
            {
                "run_dir": run_dir,
                "config": config,
                "forecasts": forecasts,
                "outcomes": outcomes,
                "prompt": str(config.get("forecast_prompt") or ""),
            }
        )
    return groups


def _load_forecast_only_groups(
    live_root: Path,
    outcome_bank: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not live_root.exists():
        return groups
    for forecast_path in sorted(live_root.glob("*/forecasts.jsonl")):
        run_dir = forecast_path.parent
        config = load_run_config(run_dir)
        forecasts = read_jsonl(forecast_path)
        model = str(config.get("model") or _first_value(forecasts, "model") or run_dir.name)
        suite = infer_suite_from_artifact_dir(run_dir, config) or str(config.get("suite_name") or "paper_forecast_stability")
        task_ids = {str(row.get("task_id")) for row in forecasts if row.get("task_id")}
        outcomes: list[dict[str, Any]] = []
        for task_id in sorted(task_ids):
            outcomes.extend(outcome_bank.get((model, task_id), []))
        groups[(suite, model)].append(
            {
                "run_dir": run_dir,
                "config": config,
                "forecasts": forecasts,
                "outcomes": outcomes,
                "prompt": str(config.get("forecast_prompt") or (config.get("metadata") or {}).get("prompt_variant_path") or ""),
            }
        )
    return groups


def _load_outcome_bank(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    bank: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not root.exists():
        return bank
    for outcomes_path in sorted(root.rglob("outcomes.jsonl")):
        run_dir = outcomes_path.parent
        config = load_run_config(run_dir)
        model = str(config.get("model") or run_dir.name)
        for row in read_jsonl(outcomes_path):
            task_id = row.get("task_id")
            if not task_id:
                continue
            row_model = str(row.get("model") or model)
            bank[(row_model, str(task_id))].append(row)
    return bank


def _score_group(suite: str, model: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    curves_by_run = [forecast_curves(run["forecasts"]) for run in runs]
    by_task: dict[str, list[dict[int, float]]] = defaultdict(list)
    for curves in curves_by_run:
        for task_id, curve in curves.items():
            by_task[task_id].append(curve)
    repeated = {task_id: curves for task_id, curves in by_task.items() if len(curves) >= 2}
    probability_stds: list[float] = []
    agreements: list[float] = []
    change_flags: list[bool] = []
    rank_correlations: list[float] = []
    for curves in repeated.values():
        budgets = sorted(set().union(*(set(curve) for curve in curves)))
        for budget in budgets:
            values = [float(curve[budget]) for curve in curves if budget in curve]
            if len(values) >= 2:
                probability_stds.append(pstdev(values))
        selected = [_selected_budget(curve) for curve in curves if curve]
        if selected:
            mode_count = max(selected.count(value) for value in set(selected))
            agreements.append(mode_count / len(selected))
            change_flags.append(len(set(selected)) > 1)
        for left_index in range(len(curves)):
            for right_index in range(left_index + 1, len(curves)):
                corr = _rank_correlation(curves[left_index], curves[right_index])
                if corr is not None:
                    rank_correlations.append(corr)
    briers: list[float] = []
    eces: list[float] = []
    for run in runs:
        outcomes = outcomes_by_task(run["outcomes"])
        curves = forecast_curves(run["forecasts"])
        probs, labels = _probability_label_pairs(curves, outcomes)
        if probs:
            briers.append(float(brier_score(probs, labels)))
            eces.append(float(expected_calibration_error(probs, labels)))
    prompt_variants = sorted({run["prompt"] for run in runs if run["prompt"]})
    note = (
        "forecast-only prompt sensitivity run paired with frozen heldout outcomes"
        if suite == "paper_forecast_stability"
        else "repeat forecasts from frozen repeatability artifacts"
    )
    return {
        "suite": suite,
        "model": model,
        "forecast_groups": len(runs),
        "tasks_with_repeats": len(repeated),
        "mean_probability_std": _fmt_or_na(sum(probability_stds) / len(probability_stds) if probability_stds else None),
        "selected_budget_agreement": _fmt_or_na(sum(agreements) / len(agreements) if agreements else None),
        "curve_rank_correlation": _fmt_or_na(sum(rank_correlations) / len(rank_correlations) if rank_correlations else None),
        "brier_std": _fmt_or_na(pstdev(briers) if len(briers) >= 2 else None),
        "ece_std": _fmt_or_na(pstdev(eces) if len(eces) >= 2 else None),
        "budget_choice_change_fraction": _fmt_or_na(sum(change_flags) / len(change_flags) if change_flags else None),
        "prompt_variants_observed": len(prompt_variants),
        "notes": note,
    }


def _plot_stability(rows: list[dict[str, Any]], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    display_rows = [row for row in rows if row.get("suite") == "paper_forecast_stability"] or rows
    labels = [_short_model(row["model"]) for row in display_rows]
    stds = [_float_or_zero(row["mean_probability_std"]) for row in display_rows]
    changes = [_float_or_zero(row["budget_choice_change_fraction"]) for row in display_rows]
    axes[0].barh(np.arange(len(labels)), stds, color="#2563eb")
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].set_xlabel("Mean std. dev. of p(success | B)")
    axes[0].set_title("Forecast Probability Stability", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.35)
    axes[1].barh(np.arange(len(labels)), changes, color="#b45309")
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Fraction with changed selected budget")
    axes[1].set_title("Budget Choice Sensitivity", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.35)
    if not display_rows:
        for ax in axes:
            ax.text(0.5, 0.5, "No repeatability artifacts found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle("Forecast Stability and Prompt Sensitivity", fontsize=13.2, fontweight="bold")
    fig.tight_layout()
    paths = [output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return paths


def _selected_budget(curve: dict[int, float]) -> int:
    for budget, probability in sorted(curve.items()):
        if probability >= 0.5:
            return int(budget)
    return int(max(curve))


def _rank_correlation(left: dict[int, float], right: dict[int, float]) -> float | None:
    budgets = sorted(set(left) & set(right))
    if len(budgets) < 2:
        return None
    left_ranks = _ranks([left[budget] for budget in budgets])
    right_ranks = _ranks([right[budget] for budget in budgets])
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return np.array(ranks, dtype=float)


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


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key) is not None:
            return row[key]
    return None


def _short_model(model: str) -> str:
    return (
        str(model)
        .replace("gemini-2.0-", "Gemini ")
        .replace("-001", "")
        .replace("DeepSeek-V3-0324", "DeepSeek V3")
        .replace("claude-3-haiku", "Claude Haiku")
        .replace("_", " ")
    )


def _fmt_or_na(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.6f}"


def _float_or_zero(value: Any) -> float:
    if value in {None, "", "NA"}:
        return 0.0
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in TABLE_FIELDS} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze forecast repeatability and prompt-sensitivity artifacts.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--live-run-root", default=None)
    parser.add_argument("--output-table", default="reports/tables/paper_table14_forecast_stability.csv")
    parser.add_argument("--output-figure-prefix", default="reports/figures/appendix_forecast_stability")
    args = parser.parse_args()
    outputs = analyze_forecast_stability(
        artifact_root=args.artifact_root,
        live_run_root=args.live_run_root,
        output_table=args.output_table,
        output_figure_prefix=args.output_figure_prefix,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
