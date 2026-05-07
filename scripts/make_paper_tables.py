#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import (
    ci_string,
    forecast_curves,
    forecast_medians,
    load_paper_runs,
    outcomes_by_task,
    score_curve_set,
)
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl
from budget2success.utils.manifest import sha256_file

try:
    from scripts.run_allocation_frontier import (
        run_allocation_frontier,
        write_fixed_budget_scheduling_table,
        write_replacement_fixed_budget_scheduling_table,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_allocation_frontier import (
        run_allocation_frontier,
        write_fixed_budget_scheduling_table,
        write_replacement_fixed_budget_scheduling_table,
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
    from scripts.run_learned_calibration_baseline import run_learned_calibration_baseline
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_learned_calibration_baseline import run_learned_calibration_baseline


TABLE_DIR = Path("reports/tables")


def make_paper_tables(
    *,
    suite: str | None = None,
    table_dir: str | Path = TABLE_DIR,
    artifact_root: str | Path | None = "reports/artifacts",
    include_artifacts: bool = True,
    corrected_artifact_root: str | Path | None = None,
    official_artifact_roots: Iterable[str | Path] | None = None,
    math_label_mode: str = "original",
    final_paper_mode: bool = False,
) -> list[Path]:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    runs = load_paper_runs(
        suite=suite,
        run_root=Path(artifact_root) / "__no_reports_runs__" if artifact_root is not None else "reports/runs",
        artifact_root=artifact_root,
        include_artifacts=include_artifacts,
        corrected_artifact_root=corrected_artifact_root,
        official_artifact_roots=_official_roots(official_artifact_roots),
        math_label_mode=math_label_mode,
    )
    runs = _paper_evidence_runs(runs)
    verifier_rows = _verifier_artifact_rows(runs)
    table3a, table3b, legacy_table3 = _table3_rows(runs, suite)
    outputs = [
        _write_csv(table_dir / "paper_table1_related_work.csv", _table1_rows()),
        _write_csv(table_dir / "paper_table2_dataset_composition.csv", _table2_rows()),
        _write_csv(table_dir / "paper_table_clean_evidence_scope.csv", _clean_evidence_scope_rows()),
        _write_csv(table_dir / "paper_table3a_calibration_capability.csv", table3a),
        _write_csv(table_dir / "paper_table3b_allocation_diagnostics.csv", table3b),
        _write_csv(table_dir / "paper_table3_main_metrics.csv", legacy_table3),
        _write_csv(table_dir / "paper_table4_calibration_split_baselines.csv", _baseline_rows("calibration")),
        _write_csv(table_dir / "paper_table4_main_baseline_summary.csv", _compressed_baseline_rows()),
        _write_csv(table_dir / "paper_table4_deployable_baselines.csv", _baseline_rows("calibration")),
        _write_csv(table_dir / "paper_table5_diagnostic_baselines.csv", _baseline_rows("diagnostic")),
        _write_csv(table_dir / "paper_table4_baselines.csv", _baseline_rows(None)),
        _write_csv(table_dir / "paper_table6_cost_runtime.csv", _table6_cost_rows(runs, final_paper_mode=final_paper_mode)),
        _write_csv(table_dir / "paper_table7_verifier_robustness.csv", _table7_verifier_robustness_rows()),
        _write_csv(table_dir / "paper_table_math_answer_type_audit.csv", _math_answer_type_audit_rows()),
        _write_csv(
            table_dir / "paper_table_manual_math_audit_summary.csv",
            _manual_math_audit_summary_rows(final_paper_mode=final_paper_mode),
        ),
        _write_csv(table_dir / "paper_table7_verifier_artifacts.csv", verifier_rows),
        _write_csv(table_dir / "paper_table5_verifier_artifacts.csv", verifier_rows),
        _write_csv(table_dir / "paper_table8_metric_definitions.csv", _metric_definition_rows()),
        _write_csv(table_dir / "paper_table_repeatability_audit.csv", _repeatability_audit_rows()),
        _write_csv(table_dir / "paper_table10_repeatability.csv", _repeatability_audit_rows()),
        _write_csv(table_dir / "paper_table11_fresh_coding.csv", _fresh_coding_rows(_official_roots(official_artifact_roots))),
        _write_csv(table_dir / "paper_table11b_fresh_coding_200.csv", _fresh_coding_200_rows()),
        _write_csv(table_dir / "paper_table11c_fresh_coding_300.csv", _fresh_coding_300_rows()),
        run_allocation_frontier(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            output_table=table_dir / "paper_table12_allocation_frontier.csv",
            figures_dir="reports/figures",
            write_figures=False,
        )[0],
        write_fixed_budget_scheduling_table(
            frontier_table=table_dir / "paper_table12_allocation_frontier.csv",
            output_table=table_dir / "paper_table15_fixed_budget_scheduling.csv",
            figures_dir="reports/figures",
        )[0],
        analyze_token_usage_proxy(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            output_table=table_dir / "paper_table13_token_usage_proxy.csv",
            output_figure_prefix="reports/figures/paper_figure10_token_usage_proxy_vs_success",
            write_figure=False,
        )[0],
        analyze_token_usage_proxy(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            dual_forecast_root="reports/runs/paper_dual_success_usage_forecast_300",
            output_table=table_dir / "paper_table13b_token_usage_proxy_300.csv",
            output_figure_prefix="reports/figures/paper_figure10b_token_usage_proxy_300",
            write_figure=False,
        )[0],
        analyze_forecast_stability(
            artifact_root=artifact_root or "reports/artifacts",
            output_table=table_dir / "paper_table14_forecast_stability.csv",
            output_figure_prefix="reports/figures/appendix_forecast_stability",
        )[0],
        run_learned_calibration_baseline(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            output_table=table_dir / "paper_table16_learned_calibration_baseline.csv",
            output_figure_prefix="reports/figures/paper_figure12_learned_calibration_baseline",
            write_figure=False,
            n_bootstrap=250,
            suites=[
                "paper_math_core",
                "paper_evalplus_humaneval_full",
                "paper_evalplus_mbpp_full",
                "paper_canitedit_descriptive",
                "paper_bigcodebench_hard",
            ],
        )[0],
        _write_csv(table_dir / "appendix_swe_official_mini.csv", _swe_official_mini_rows(official_artifact_roots)),
        _write_csv(
            table_dir / "paper_table18_bigcodebench_hard.csv",
            _bigcodebench_hard_rows(artifact_root, official_artifact_roots),
        ),
        _write_csv(table_dir / "paper_table19_canitedit_descriptive.csv", _canitedit_rows(artifact_root)),
        analyze_token_usage_proxy(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            output_table=table_dir / "paper_table20_replacement_token_usage_proxy.csv",
            output_figure_prefix="reports/figures/paper_figure13_replacement_token_proxy_vs_success",
            write_figure=False,
            suite_filter={"paper_bigcodebench_hard", "paper_canitedit_descriptive"},
        )[0],
        run_allocation_frontier(
            artifact_root=artifact_root or "reports/artifacts",
            split_dir="reports/splits",
            output_table=table_dir / "paper_table21_replacement_allocation_frontier_raw.csv",
            figures_dir="reports/figures",
            write_figures=False,
            suite_filter={"paper_bigcodebench_hard", "paper_canitedit_descriptive"},
        )[0],
        write_replacement_fixed_budget_scheduling_table(
            frontier_table=table_dir / "paper_table21_replacement_allocation_frontier_raw.csv",
            output_table=table_dir / "paper_table21_replacement_fixed_budget_scheduling.csv",
            figures_dir="reports/figures",
        )[0],
        _write_csv(table_dir / "appendix_agentic_bridge.csv", _agentic_bridge_rows()),
        _write_csv(table_dir / "paper_table9_release_checklist.csv", _release_checklist_rows()),
    ]
    _write_artifact_checklist_doc(table_dir / "paper_table7_verifier_artifacts.csv")
    _assert_no_forbidden_main_table_rows(outputs)
    return outputs


def _table1_rows() -> list[dict[str, str]]:
    return [
        {
            "work_or_benchmark": "GSM8K / MATH",
            "target": "math answer correctness",
            "pre_execution_forecast": "no",
            "verified_success": "yes",
            "token_budget_curve": "no",
            "multi_domain": "no",
            "how_tokencapbench_differs": "adds calibrated forecasts over hard generated-token budgets",
        },
        {
            "work_or_benchmark": "EvalPlus / coding unit tests",
            "target": "program correctness",
            "pre_execution_forecast": "no",
            "verified_success": "yes",
            "token_budget_curve": "no",
            "multi_domain": "coding",
            "how_tokencapbench_differs": "measures probability of verified pass at each cap before execution",
        },
        {
            "work_or_benchmark": "SWE-bench",
            "target": "repository issue resolution",
            "pre_execution_forecast": "no",
            "verified_success": "yes",
            "token_budget_curve": "no",
            "multi_domain": "software engineering",
            "how_tokencapbench_differs": "uses SWE as a verifier-backed substrate rather than a new task source",
        },
        {
            "work_or_benchmark": "Raw length prediction",
            "target": "response length",
            "pre_execution_forecast": "yes",
            "verified_success": "no",
            "token_budget_curve": "sometimes",
            "multi_domain": "varies",
            "how_tokencapbench_differs": "target is verified success under budget, not generated length",
        },
    ]


def _clean_evidence_scope_rows() -> list[dict[str, str]]:
    return [
        {
            "track": "math",
            "source": "GSM8K + MATH",
            "tasks": "1000",
            "models": "4",
            "verifier": "numeric/symbolic",
            "role": "core evidence",
        },
        {
            "track": "coding",
            "source": "HumanEval+ + MBPP+",
            "tasks": "542",
            "models": "4",
            "verifier": "EvalPlus",
            "role": "core evidence",
        },
        {
            "track": "hard coding",
            "source": "BigCodeBench-Hard",
            "tasks": "148",
            "models": "5",
            "verifier": "official BigCodeBench package",
            "role": "main extension",
        },
        {
            "track": "code editing",
            "source": "CanItEdit",
            "tasks": "105",
            "models": "4",
            "verifier": "provided tests",
            "role": "editing bridge",
        },
        {
            "track": "fresh coding",
            "source": "LiveCodeBench-300",
            "tasks": "300",
            "models": "2",
            "verifier": "official LiveCodeBench labels",
            "role": "appendix freshness",
        },
        {
            "track": "stability",
            "source": "prompt variants",
            "tasks": "150",
            "models": "2+",
            "verifier": "frozen verifier outcomes",
            "role": "appendix stability",
        },
    ]


INCOMPATIBLE_MAIN_TABLE_PATTERNS = (
    "swebench",
    "swe_verified",
    "bfcl",
    "bugsinpy",
    "repoexec",
    "realbench",
    "openhands",
    "aider",
)

FORBIDDEN_MAIN_TABLE_VALUES = {
    "mock-model",
    "official_labels_absent",
    "failed_or_incomplete",
    "swebench",
    "swe_verified",
    "bfcl",
    "aider_polyglot",
}


def _paper_evidence_runs(runs: Iterable[Any]) -> list[Any]:
    return [run for run in runs if str(getattr(run, "model", "") or "") != "mock-model"]


def _truthy_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _exclude_from_dataset_composition(task: TaskRecord, split: str) -> bool:
    metadata = task.metadata or {}
    blob = " ".join(
        [
            str(task.track),
            str(task.source),
            str(task.verifier),
            split,
            str(metadata.get("paper_role") or ""),
        ]
    ).lower()
    if any(pattern in blob for pattern in INCOMPATIBLE_MAIN_TABLE_PATTERNS):
        return not (
            metadata.get("official_harness_status") == "completed"
            and _truthy_metadata(metadata.get("chat_completion_compatible"))
        )
    return str(metadata.get("paper_role") or "").lower() in {"future_work", "infrastructure_only"}


def _table2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_paths = (
        sorted(Path("data/processed").glob("paper_*.jsonl"))
        + sorted(Path("data/tasks").glob("paper_*.jsonl"))
        + sorted(Path("data/processed").glob("swe_verified_smoke.jsonl"))
    )
    for path in task_paths:
        if not path.exists():
            continue
        grouped: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        for raw in read_jsonl(path):
            task = TaskRecord.model_validate(raw)
            if _exclude_from_dataset_composition(task, path.stem):
                continue
            grid = ",".join(str(value) for value in (task.budget_grid or []))
            role = str(task.metadata.get("paper_role") or "")
            grouped[(task.track, task.source, task.verifier, grid, role)] += 1
        for (track, source, verifier, grid, role), count in sorted(grouped.items()):
            rows.append(
                {
                    "track": track,
                    "source": source,
                    "tasks": count,
                    "budget_grid": grid,
                    "verifier": verifier,
                    "split": path.stem,
                    "paper_role": role,
                }
            )
    return rows


def _table3_rows(runs, suite: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap = _read_csv(TABLE_DIR / "bootstrap_main_metrics.csv")
    run_costs = {
        (run.suite or suite or "", run.model, run.run_id): _cost_for_run(run)
        for run in runs
    }
    run_sources = {
        (run.suite or suite or "", run.model, run.run_id): run.artifact_source
        for run in runs
    }
    if bootstrap:
        filtered = [row for row in bootstrap if row.get("track") == "all" and row.get("source") == "all"]
        if suite:
            filtered = [row for row in filtered if row.get("suite") == suite]
        by_run: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in filtered:
            by_run[(row.get("suite", ""), row["model"], row["run_id"])][row["metric"]] = row
        legacy_rows = []
        table3a = []
        table3b = []
        for (run_suite, model, run_id), metrics in sorted(by_run.items()):
            common = {"suite": run_suite, "model": model}
            table3a.append(
                {
                    **common,
                    "brier_ci": _ci(metrics, "brier"),
                    "ece_ci": _ci(metrics, "ece"),
                    "success_at_max_budget_ci": _ci(metrics, "success_at_max_budget"),
                    "censoring_rate_ci": _ci(metrics, "censoring_rate"),
                    "ranking_accuracy_ci": _ci(metrics, "task_budget_ranking_accuracy"),
                    "estimated_cost_usd": run_costs.get((run_suite, model, run_id), ""),
                }
            )
            table3b.append(
                {
                    **common,
                    "absolute_log_budget_error_ci": _ci(metrics, "absolute_log_budget_error_mean"),
                    "signed_log_budget_error_ci": _ci(metrics, "signed_log_budget_error_mean"),
                    "underbudget_rate_ci": _ci(metrics, "underbudget_rate"),
                    "overbudget_rate_ci": _ci(metrics, "overbudget_rate"),
                    "regret_ci": _ci(metrics, "regret"),
                    "normalized_regret_ci": _ci(metrics, "normalized_regret"),
                    "truncation_rate_ci": _ci(metrics, "truncation_rate"),
                }
            )
            legacy_rows.append(
                {
                    **common,
                    "run_id": run_id,
                    "track": "all",
                    "source": "all",
                    "brier_ci": table3a[-1]["brier_ci"],
                    "ece_ci": table3a[-1]["ece_ci"],
                    "success_at_max_budget_ci": table3a[-1]["success_at_max_budget_ci"],
                    "censoring_rate_ci": table3a[-1]["censoring_rate_ci"],
                    "absolute_log_budget_error_ci": table3b[-1]["absolute_log_budget_error_ci"],
                    "signed_log_budget_error_ci": table3b[-1]["signed_log_budget_error_ci"],
                    "underbudget_rate_ci": table3b[-1]["underbudget_rate_ci"],
                    "overbudget_rate_ci": table3b[-1]["overbudget_rate_ci"],
                    "regret_ci": table3b[-1]["regret_ci"],
                    "normalized_regret_ci": table3b[-1]["normalized_regret_ci"],
                    "forecast_monotonicity_violation_rate_ci": _ci(metrics, "forecast_monotonicity_violation_rate"),
                    "outcome_nonmonotonicity_rate_ci": _ci(metrics, "outcome_nonmonotonicity_rate"),
                    "task_budget_ranking_accuracy_ci": table3a[-1]["ranking_accuracy_ci"],
                    "truncation_rate_ci": table3b[-1]["truncation_rate_ci"],
                    "estimated_cost_usd": table3a[-1]["estimated_cost_usd"],
                    "artifact_source": run_sources.get((run_suite, model, run_id), ""),
                }
            )
        return table3a, table3b, legacy_rows
    table3a = []
    table3b = []
    legacy_rows = []
    for run in runs:
        run_suite = run.suite or suite or ""
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        common = {"suite": run_suite, "model": run.model}
        table3a.append(
            {
                **common,
                "brier_ci": _point_ci(scored.get("brier")),
                "ece_ci": _point_ci(scored.get("ece")),
                "success_at_max_budget_ci": _point_ci(scored.get("success_at_max_budget")),
                "censoring_rate_ci": _point_ci(scored.get("censoring_rate")),
                "ranking_accuracy_ci": _point_ci(scored.get("task_budget_ranking_accuracy")),
                "estimated_cost_usd": _cost_for_run(run),
            }
        )
        table3b.append(
            {
                **common,
                "absolute_log_budget_error_ci": _point_ci(scored.get("absolute_log_budget_error_mean")),
                "signed_log_budget_error_ci": _point_ci(scored.get("signed_log_budget_error_mean")),
                "underbudget_rate_ci": _point_ci(scored.get("underbudget_rate")),
                "overbudget_rate_ci": _point_ci(scored.get("overbudget_rate")),
                "regret_ci": _point_ci(scored.get("regret")),
                "normalized_regret_ci": _point_ci(scored.get("normalized_regret")),
                "truncation_rate_ci": _point_ci(scored.get("truncation_rate")),
            }
        )
        legacy_rows.append(
            {
                **common,
                "run_id": run.run_id,
                "track": "all",
                "source": "all",
                "brier_ci": table3a[-1]["brier_ci"],
                "ece_ci": table3a[-1]["ece_ci"],
                "success_at_max_budget_ci": table3a[-1]["success_at_max_budget_ci"],
                "censoring_rate_ci": table3a[-1]["censoring_rate_ci"],
                "absolute_log_budget_error_ci": table3b[-1]["absolute_log_budget_error_ci"],
                "signed_log_budget_error_ci": table3b[-1]["signed_log_budget_error_ci"],
                "underbudget_rate_ci": table3b[-1]["underbudget_rate_ci"],
                "overbudget_rate_ci": table3b[-1]["overbudget_rate_ci"],
                "regret_ci": table3b[-1]["regret_ci"],
                "normalized_regret_ci": table3b[-1]["normalized_regret_ci"],
                "forecast_monotonicity_violation_rate_ci": _point_ci(scored.get("forecast_monotonicity_violation_rate")),
                "outcome_nonmonotonicity_rate_ci": _point_ci(scored.get("outcome_nonmonotonicity_rate")),
                "task_budget_ranking_accuracy_ci": table3a[-1]["ranking_accuracy_ci"],
                "truncation_rate_ci": table3b[-1]["truncation_rate_ci"],
                "estimated_cost_usd": table3a[-1]["estimated_cost_usd"],
                "artifact_source": run.artifact_source,
            }
        )
    return table3a, table3b, legacy_rows


def _baseline_rows(baseline_class: str | None) -> list[dict[str, Any]]:
    if baseline_class == "calibration":
        ready = _read_csv(TABLE_DIR / "paper_table4_calibration_split_baselines.csv")
        if ready:
            return ready
    if baseline_class == "diagnostic":
        ready = _read_csv(TABLE_DIR / "paper_table5_diagnostic_baselines.csv")
        if ready:
            return ready
    baseline = _read_csv(TABLE_DIR / "baseline_comparison.csv")
    rows = []
    for row in baseline:
        row_class = row.get("baseline_class") or ("posthoc_diagnostic" if row.get("forecast_method") == "output_length_proxy" else "model_forecast_raw")
        if baseline_class == "calibration" and row_class not in {
            "model_forecast_raw",
            "model_forecast_recalibrated",
            "calibration_split_baseline",
        }:
            continue
        if baseline_class == "diagnostic" and row_class not in {"test_distribution_diagnostic", "posthoc_diagnostic"}:
            continue
        if baseline_class not in {None, "calibration", "diagnostic"} and row_class != baseline_class:
            continue
        rows.append(
            {
                "suite": row.get("suite"),
                "model": row.get("model"),
                "forecast_method": row.get("forecast_method"),
                "baseline_class": row_class,
                "n_eval_tasks": row.get("n_eval_tasks") or row.get("n_tasks"),
                "brier": row.get("brier"),
                "ece": row.get("ece"),
                "regret": row.get("regret"),
                "brier_low": row.get("brier_low") or row.get("brier_ci_low", ""),
                "brier_high": row.get("brier_high") or row.get("brier_ci_high", ""),
                "ece_low": row.get("ece_low") or row.get("ece_ci_low", ""),
                "ece_high": row.get("ece_high") or row.get("ece_ci_high", ""),
                "regret_low": row.get("regret_low") or row.get("regret_ci_low", ""),
                "regret_high": row.get("regret_high") or row.get("regret_ci_high", ""),
                "brier_ci": row.get("brier_ci") or _point_ci(row.get("brier")),
                "ece_ci": row.get("ece_ci") or _point_ci(row.get("ece")),
                "regret_ci": row.get("regret_ci") or _point_ci(row.get("regret")),
                "n_bootstrap": row.get("n_bootstrap", ""),
                "notes": row.get("notes"),
            }
        )
    return rows


def _compressed_baseline_rows() -> list[dict[str, Any]]:
    rows = _baseline_rows("calibration")
    if not rows:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("track") not in {"", None, "all"} or row.get("source") not in {"", None, "all"}:
            continue
        grouped[(str(row.get("suite") or ""), str(row.get("model") or ""))].append(row)
    output: list[dict[str, Any]] = []
    for (suite, model), group_rows in sorted(grouped.items()):
        raw = _first_method(group_rows, lambda row: row.get("forecast_method") == "self_forecast_raw")
        recalibrated = _best_by_metric(
            group_rows,
            "brier",
            lambda row: str(row.get("baseline_class") or "") == "model_forecast_recalibrated"
            or "recalibrated" in str(row.get("forecast_method") or ""),
        )
        simple_prior = _best_by_metric(
            group_rows,
            "brier",
            lambda row: str(row.get("baseline_class") or "") == "calibration_split_baseline"
            and row.get("forecast_method") != "single_budget_midpoint",
        )
        regret = _best_by_metric(group_rows, "regret", lambda row: True)
        output.append(
            {
                "suite": suite,
                "model": model,
                "raw_self_forecast_brier_ci": _row_ci(raw, "brier"),
                "best_recalibrated_method": recalibrated.get("forecast_method", "") if recalibrated else "",
                "best_recalibrated_brier_ci": _row_ci(recalibrated, "brier"),
                "best_simple_prior_method": simple_prior.get("forecast_method", "") if simple_prior else "",
                "best_simple_prior_brier_ci": _row_ci(simple_prior, "brier"),
                "best_regret_method": regret.get("forecast_method", "") if regret else "",
                "best_regret_ci": _row_ci(regret, "regret"),
            }
        )
    return output


def _first_method(rows: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def _best_by_metric(rows: list[dict[str, Any]], metric: str, predicate) -> dict[str, Any] | None:
    candidates = [row for row in rows if predicate(row) and row.get(metric) not in {"", None}]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[metric]))


def _row_ci(row: dict[str, Any] | None, metric: str) -> str:
    if not row:
        return ""
    return str(row.get(f"{metric}_ci") or _point_ci(row.get(metric)))


def _table7_verifier_robustness_rows() -> list[dict[str, Any]]:
    rows = _read_csv(TABLE_DIR / "math_verifier_delta_summary.csv")
    if not rows:
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "suite": row.get("suite"),
                "source": row.get("source"),
                "verifier_policy": row.get("verifier_policy") or row.get("verifier_mode") or "",
                "math_verify_available": row.get("math_verify_available", ""),
                "unsupported_rows": row.get("unsupported_rows", "0"),
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "n_rows": row.get("n_rows"),
                "n_changed": row.get("n_changed"),
                "change_rate": row.get("change_rate"),
                "old_success_rate": row.get("old_success_rate"),
                "new_success_rate": row.get("new_success_rate"),
                "success_delta": row.get("success_delta"),
                "verifier_mode": row.get("verifier_mode") or row.get("mode") or "strict",
            }
        )
    return result


def _math_answer_type_audit_rows() -> list[dict[str, Any]]:
    rows = _read_csv(TABLE_DIR / "math_reverification_audit.csv")
    if not rows:
        return [
            {
                "suite": "paper_math_core",
                "source": "gsm8k_numeric_and_hendrycks_math_symbolic",
                "answer_type": "mixed",
                "model": "all",
                "n_rows": 20000,
                "n_changed": 0,
                "change_rate": 0.0,
                "old_success_rate": "",
                "new_success_rate": "",
                "success_delta": 0.0,
                "verifier_mode": "frozen_paper_artifacts",
                "notes": "Frozen paper artifact labels are used; no fresh math answer-type re-verification was rerun in this packaging pass.",
            }
        ]
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"n_rows": 0, "n_changed": 0, "old_success": 0, "new_success": 0, "verifier_modes": set()}
    )
    for row in rows:
        key = (
            row.get("suite", ""),
            row.get("source", ""),
            row.get("answer_type", "") or "unknown",
            row.get("model", ""),
        )
        group = grouped[key]
        group["n_rows"] += 1
        changed = _truthy(row.get("changed"))
        old_success = _truthy(row.get("recorded_success"))
        new_success = _truthy(row.get("reverified_success"))
        group["n_changed"] += 1 if changed else 0
        group["old_success"] += 1 if old_success else 0
        group["new_success"] += 1 if new_success else 0
        group["verifier_modes"].add(row.get("verifier_selected") or row.get("mode") or "")
    result: list[dict[str, Any]] = []
    for (suite, source, answer_type, model), values in sorted(grouped.items()):
        n_rows = int(values["n_rows"])
        old_rate = values["old_success"] / n_rows if n_rows else 0.0
        new_rate = values["new_success"] / n_rows if n_rows else 0.0
        result.append(
            {
                "suite": suite,
                "source": source,
                "answer_type": answer_type,
                "model": model,
                "n_rows": n_rows,
                "n_changed": int(values["n_changed"]),
                "change_rate": int(values["n_changed"]) / n_rows if n_rows else 0.0,
                "old_success_rate": old_rate,
                "new_success_rate": new_rate,
                "success_delta": new_rate - old_rate,
                "verifier_mode": ";".join(sorted(value for value in values["verifier_modes"] if value)),
            }
        )
    return result


def _manual_math_audit_summary_rows(*, final_paper_mode: bool = False) -> list[dict[str, Any]]:
    path = TABLE_DIR / "manual_math_verifier_audit_sample.csv"
    rows = _read_csv(path)
    if not rows:
        if final_paper_mode:
            raise ValueError("Final-paper mode requires a nonempty manual math verifier audit sample.")
        return []
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"sampled": 0, "annotated": 0, "labels": defaultdict(int)})
    for row in rows:
        source = row.get("source") or "unknown"
        label = str(row.get("human_audit_label") or "").strip()
        grouped[source]["sampled"] += 1
        if label:
            grouped[source]["annotated"] += 1
            grouped[source]["labels"][label] += 1
    total_annotated = sum(int(values["annotated"]) for values in grouped.values())
    if final_paper_mode and total_annotated == 0:
        raise ValueError("Final-paper mode requires at least one annotated manual math verifier audit row.")
    return [
        {
            "source": source,
            "n_sampled": values["sampled"],
            "n_annotated": values["annotated"],
            "human_audit_label_counts": json.dumps(dict(sorted(values["labels"].items())), sort_keys=True),
            "notes": (
                "WARNING: no manual audit labels are annotated."
                if int(values["annotated"]) == 0
                else "human_audit_label audits whether the current automatic verifier decision matches manual review."
            ),
        }
        for source, values in sorted(grouped.items())
    ]


def _repeatability_audit_rows() -> list[dict[str, Any]]:
    try:
        runs = load_paper_runs(suite="paper_repeatability_small", artifact_root="reports/artifacts", include_artifacts=True)
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str, str, int], dict[str, dict[str, list[bool]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_task_repeat: dict[tuple[str, str, str, str, str], dict[int, bool]] = defaultdict(dict)
    for run in runs:
        for outcome in run.outcomes:
            metadata = outcome.get("metadata") or {}
            repeat_id = metadata.get("repeat_id")
            if repeat_id is None:
                continue
            source = str(metadata.get("source") or outcome.get("source") or "unknown")
            task_id = str(outcome.get("task_id"))
            budget = int(outcome.get("budget"))
            success = bool(outcome.get("success"))
            suite = run.suite or "paper_repeatability_small"
            model = run.model
            by_group[(suite, model, source, budget)][task_id][str(repeat_id)].append(success)
            by_task_repeat[(suite, model, source, task_id, str(repeat_id))][budget] = success
    if not by_group:
        return []
    nonmonotonic_by_source = _repeatability_nonmonotonicity(by_task_repeat)
    for (suite, model, source, budget), task_map in sorted(by_group.items()):
        task_agreements: list[float] = []
        success_values: list[bool] = []
        repeat_success_rates: dict[str, list[bool]] = defaultdict(list)
        repeat_counts: list[int] = []
        for repeat_map in task_map.values():
            task_repeat_values = [bool(values[-1]) for values in repeat_map.values() if values]
            if not task_repeat_values:
                continue
            repeat_counts.append(len(task_repeat_values))
            task_agreements.append(_agreement_rate(task_repeat_values))
            success_values.extend(task_repeat_values)
            for repeat_id, values in repeat_map.items():
                if values:
                    repeat_success_rates[repeat_id].append(bool(values[-1]))
        per_repeat_rates = [
            sum(1.0 for value in values if value) / len(values)
            for values in repeat_success_rates.values()
            if values
        ]
        rows.append(
            {
                "suite": suite,
                "model": model,
                "source": source,
                "budget": budget,
                "n_tasks": len(task_map),
                "n_repeats": max(repeat_counts) if repeat_counts else 0,
                "success_agreement_rate": sum(task_agreements) / len(task_agreements) if task_agreements else "",
                "mean_success_rate": sum(1.0 for value in success_values if value) / len(success_values) if success_values else "",
                "std_success_rate": pstdev(per_repeat_rates) if len(per_repeat_rates) > 1 else 0.0,
                "nonmonotonicity_rate": nonmonotonic_by_source.get((suite, model, source), ""),
            }
        )
    return rows


def _fresh_coding_rows(official_artifact_roots: Iterable[str | Path] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = []
    for root in _official_roots(official_artifact_roots):
        try:
            runs.extend(
                load_paper_runs(
                    suite="paper_livecodebench_fresh_small",
                    artifact_root=root,
                    include_artifacts=True,
                    official_artifact_roots=[root],
                )
            )
        except FileNotFoundError:
            continue
    for run in runs:
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        if not scored.get("n_tasks"):
            continue
        rows.append(
            {
                "suite": run.suite or "paper_livecodebench_fresh_small",
                "model": run.model,
                "n_tasks": scored.get("n_tasks", ""),
                "brier": scored.get("brier", ""),
                "ece": scored.get("ece", ""),
                "success_at_max_budget": scored.get("success_at_max_budget", ""),
                "regret": scored.get("regret", ""),
                "official_harness_status": "completed",
                "notes": "Official LiveCodeBench artifact discovered.",
            }
        )
    if rows:
        return rows
    placeholders = _livecodebench_placeholder_rows()
    if placeholders:
        return placeholders
    split = Path("data/processed/paper_livecodebench_fresh_small.jsonl")
    if not split.exists():
        return []
    tasks = [TaskRecord.model_validate(row) for row in read_jsonl(split)]
    return [
        {
            "suite": "paper_livecodebench_fresh_small",
            "model": "",
            "n_tasks": len(tasks),
            "brier": "",
            "ece": "",
            "success_at_max_budget": "",
            "regret": "",
            "official_harness_status": "configured_not_run",
            "notes": "Fresh coding split is configured; no official LiveCodeBench run is included as main evidence.",
        }
    ]


def _livecodebench_placeholder_rows() -> list[dict[str, Any]]:
    artifact_root = Path("reports/artifacts")
    rows: list[dict[str, Any]] = []
    for outcomes_path in sorted(artifact_root.glob("*/outcomes.jsonl")):
        config = {}
        config_path = outcomes_path.parent / "config_snapshot.yaml"
        if config_path.exists():
            try:
                from budget2success.utils.config import load_yaml

                config = load_yaml(config_path)
            except Exception:
                config = {}
        if "paper_livecodebench_fresh_small" not in outcomes_path.parent.name and config.get("suite_name") != "paper_livecodebench_fresh_small":
            continue
        outcomes = read_jsonl(outcomes_path)
        if not any(_placeholder_unverified(row) for row in outcomes):
            continue
        task_count = len({str(row.get("task_id")) for row in outcomes if row.get("task_id")})
        rows.append(
            {
                "suite": "paper_livecodebench_fresh_small",
                "model": _first_value(outcomes, "model") or config.get("model", ""),
                "n_tasks": task_count,
                "brier": "",
                "ece": "",
                "success_at_max_budget": "",
                "regret": "",
                "official_harness_status": "placeholder_unverified",
                "notes": "Placeholder local labels are present but excluded from main metrics until official LiveCodeBench labels are merged.",
            }
        )
    return rows


def _fresh_coding_200_rows() -> list[dict[str, Any]]:
    estimate = _read_json(Path("reports/tables/paper_livecodebench_fresh_200_cost_estimate.json"))
    config = _read_yaml(Path("configs/paper_livecodebench_fresh_200.yaml"))
    task_file = Path(str(config.get("task_file") or "data/processed/paper_livecodebench_fresh_small.jsonl"))
    local_tasks = _jsonl_count(task_file)
    rows: list[dict[str, Any]] = []
    try:
        runs = load_paper_runs(
            suite="paper_livecodebench_fresh_200",
            run_root="reports/runs",
            artifact_root=None,
            include_artifacts=False,
            official_artifact_roots=_official_roots(["reports/artifacts_livecodebench_official"]),
        )
    except FileNotFoundError:
        runs = []
    for run in runs:
        if not run.outcomes:
            continue
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        official = any(
            ((row.get("verification") or {}).get("metadata") or {}).get("label_source") == "official_livecodebench"
            for row in run.outcomes
        )
        rows.append(
            {
                "suite": "paper_livecodebench_fresh_200",
                "configured_limit": config.get("limit", 200),
                "local_tasks_available": local_tasks,
                "model": run.model,
                "models": run.model,
                "n_tasks": scored.get("n_tasks", ""),
                "brier": scored.get("brier", ""),
                "ece": scored.get("ece", ""),
                "success_at_max_budget": scored.get("success_at_max_budget", ""),
                "regret": scored.get("regret", ""),
                "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
                "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 15),
                "run_status": "completed_live_api",
                "official_harness_status": "completed" if official else "local_labels_only",
                "notes": "Expanded LiveCodeBench run completed with official labels; used as appendix freshness evidence.",
            }
        )
    if rows:
        return rows
    run_root = Path("reports/runs/paper_livecodebench_fresh_200")
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    status = "completed_live_api_official_not_run" if any((path / "outcomes.jsonl").exists() for path in run_dirs) else "configured_not_run"
    note = (
        "Live forecast/solver outputs exist, but official LiveCodeBench labels were not merged."
        if status.startswith("completed")
        else "Cost-estimated under cap; run not completed in this workspace."
    )
    return [
        {
            "suite": "paper_livecodebench_fresh_200",
            "configured_limit": config.get("limit", 200),
            "local_tasks_available": local_tasks,
            "models": ",".join(estimate.get("models", [])) if estimate else "",
            "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
            "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 15),
            "run_status": status,
            "official_harness_status": "not_run",
            "notes": note,
        }
    ]


def _fresh_coding_300_rows() -> list[dict[str, Any]]:
    estimate = _read_json(Path("reports/tables/paper_livecodebench_fresh_300_cost_estimate.json"))
    config = _read_yaml(Path("configs/paper_livecodebench_fresh_300_provider.yaml"))
    task_file = Path(str(config.get("task_file") or "data/processed/paper_livecodebench_fresh_300.jsonl"))
    local_tasks = _jsonl_count(task_file)
    rows: list[dict[str, Any]] = []
    try:
        runs = load_paper_runs(
            suite="paper_livecodebench_fresh_300",
            run_root="reports/runs",
            artifact_root=None,
            include_artifacts=False,
            official_artifact_roots=_official_roots(["reports/artifacts_livecodebench_official"]),
        )
    except FileNotFoundError:
        runs = []
    for run in runs:
        if not run.outcomes:
            continue
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        official = any((row.get("metadata") or {}).get("label_source") == "official_livecodebench" for row in run.outcomes)
        rows.append(
            {
                "suite": "paper_livecodebench_fresh_300",
                "configured_limit": config.get("limit") or 300,
                "local_tasks_available": local_tasks,
                "model": run.model,
                "models": run.model,
                "n_tasks": scored.get("n_tasks", ""),
                "brier": scored.get("brier", ""),
                "ece": scored.get("ece", ""),
                "success_at_max_budget": scored.get("success_at_max_budget", ""),
                "regret": scored.get("regret", ""),
                "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
                "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 15),
                "run_status": "completed_live_api",
                "official_harness_status": "completed" if official else "local_labels_only",
                "notes": "LiveCodeBench-300 expansion completed with official labels; used as appendix freshness evidence.",
            }
        )
    if rows:
        return rows
    run_root = Path("reports/runs/paper_livecodebench_fresh_300")
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    status = "completed_live_api_official_not_run" if any((path / "outcomes.jsonl").exists() for path in run_dirs) else "configured_not_run"
    note = (
        "Live forecast/solver outputs exist, but official LiveCodeBench labels were not merged."
        if status.startswith("completed")
        else "Cost-estimated under cap; run not completed in this workspace."
    )
    return [
        {
            "suite": "paper_livecodebench_fresh_300",
            "configured_limit": config.get("limit") or 300,
            "local_tasks_available": local_tasks,
            "models": ",".join(estimate.get("models", [])) if estimate else "",
            "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
            "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 15),
            "run_status": status,
            "official_harness_status": "not_run",
            "notes": note,
        }
    ]


def _bigcodebench_hard_rows(
    artifact_root: str | Path | None,
    official_artifact_roots: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    suite = "paper_bigcodebench_hard"
    official_roots = _bigcodebench_official_roots(official_artifact_roots)
    runs = _load_replacement_runs(suite, artifact_root, extra_roots=official_roots)
    raw_outcomes = _raw_replacement_outcomes(suite, artifact_root)
    for root in official_roots:
        raw_outcomes.extend(_raw_replacement_outcomes(suite, root))
    official_labels = sum(
        1 for row in raw_outcomes if (row.get("metadata") or {}).get("label_source") == "official_bigcodebench"
    )
    config = _replacement_config("paper_bigcodebench_hard")
    task_file = Path(str(config.get("task_file") or "data/processed/paper_bigcodebench_hard.jsonl"))
    status = (
        "official_labels_completed"
        if official_labels
        else ("official_labels_absent" if raw_outcomes else ("configured_not_run" if task_file.exists() else "not_run"))
    )
    estimate = _read_json(Path("reports/tables/paper_bigcodebench_hard_cost_estimate.json"))
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.outcomes or not _run_has_official_bigcodebench_labels(run):
            continue
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        rows.append(
            {
                "suite": suite,
                "model": run.model,
                "n_tasks": scored.get("n_tasks", ""),
                "budget_grid": _budget_grid_text(run.outcomes, fallback="128,256,512,1024,2048,4096"),
                "brier": _metric_text(scored.get("brier")),
                "ece": _metric_text(scored.get("ece")),
                "success_at_max_budget": _metric_text(scored.get("success_at_max_budget")),
                "ranking_accuracy": _metric_text(scored.get("task_budget_ranking_accuracy")),
                "underbudget_rate": _metric_text(scored.get("underbudget_rate")),
                "overbudget_rate": _metric_text(scored.get("overbudget_rate")),
                "normalized_regret": _metric_text(scored.get("normalized_regret")),
                "estimated_cost_usd": _cost_for_run(run),
                "official_harness_status": status,
                "paper_role": "main_candidate" if status == "official_labels_completed" else "future_work",
            }
        )
    if rows:
        return rows
    return [
        {
            "suite": suite,
            "model": str(model),
            "n_tasks": _jsonl_count(task_file),
            "budget_grid": "128,256,512,1024,2048,4096",
            "brier": "",
            "ece": "",
            "success_at_max_budget": "",
            "ranking_accuracy": "",
            "underbudget_rate": "",
            "overbudget_rate": "",
            "normalized_regret": "",
            "estimated_cost_usd": estimate.get("estimated_total_cost_usd", ""),
            "official_harness_status": status,
            "paper_role": "future_work" if status != "official_labels_completed" else "main_candidate",
        }
        for model in _config_models(config)
    ]


def _canitedit_rows(artifact_root: str | Path | None) -> list[dict[str, Any]]:
    suite = "paper_canitedit_descriptive"
    runs = _load_replacement_runs(suite, artifact_root)
    estimate = _read_json(Path("reports/tables/paper_canitedit_descriptive_cost_estimate.json"))
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not run.outcomes:
            continue
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        label_sources = {
            str((row.get("metadata") or {}).get("label_source") or "")
            for row in run.outcomes
        }
        rows.append(
            {
                "suite": suite,
                "instruction_style": str((run.config.get("metadata") or {}).get("instruction_style") or "descriptive"),
                "model": run.model,
                "n_tasks": scored.get("n_tasks", ""),
                "budget_grid": _budget_grid_text(run.outcomes, fallback="128,256,512,1024,2048"),
                "brier": _metric_text(scored.get("brier")),
                "ece": _metric_text(scored.get("ece")),
                "success_at_max_budget": _metric_text(scored.get("success_at_max_budget")),
                "ranking_accuracy": _metric_text(scored.get("task_budget_ranking_accuracy")),
                "underbudget_rate": _metric_text(scored.get("underbudget_rate")),
                "overbudget_rate": _metric_text(scored.get("overbudget_rate")),
                "normalized_regret": _metric_text(scored.get("normalized_regret")),
                "pass_rate_by_budget": _pass_rate_by_budget_text(run.outcomes),
                "verifier_status": "completed" if all("success" in row for row in run.outcomes) else "missing_labels",
                "official_harness_status": "provided_tests_completed" if "canitedit_provided_tests" in label_sources else "configured_not_run",
                "paper_role": "main_candidate"
                if "canitedit_provided_tests" in label_sources and scored.get("n_tasks", 0) >= 100
                else "appendix_or_configured",
            }
        )
    if rows:
        return rows
    config = _replacement_config("paper_canitedit_descriptive")
    task_file = Path(str(config.get("task_file") or "data/processed/paper_canitedit_descriptive.jsonl"))
    return [
        {
            "suite": suite,
            "instruction_style": str((config.get("metadata") or {}).get("instruction_style") or "descriptive"),
            "model": str(model),
            "n_tasks": _jsonl_count(task_file),
            "budget_grid": "128,256,512,1024,2048",
            "brier": "",
            "ece": "",
            "success_at_max_budget": "",
            "ranking_accuracy": "",
            "underbudget_rate": "",
            "overbudget_rate": "",
            "normalized_regret": "",
            "pass_rate_by_budget": "",
            "verifier_status": "configured_not_run",
            "official_harness_status": "provided_tests_available" if task_file.exists() else "configured_not_run",
            "paper_role": "configured",
        }
        for model in _config_models(config)
    ]


def _load_replacement_runs(
    suite: str,
    artifact_root: str | Path | None,
    *,
    extra_roots: Iterable[str | Path] | None = None,
) -> list[Any]:
    runs: list[Any] = []
    roots = [artifact_root or "reports/artifacts", *(extra_roots or [])]
    for root in roots:
        try:
            runs.extend(
                load_paper_runs(
                    suite=suite,
                    run_root="reports/runs",
                    artifact_root=root,
                    include_artifacts=True,
                )
            )
        except FileNotFoundError:
            continue
    return _paper_evidence_runs(runs)


def _raw_replacement_outcomes(suite: str, artifact_root: str | Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_roots = [Path("reports/runs") / suite]
    if artifact_root is not None:
        candidate_roots.append(Path(artifact_root))
    for root in candidate_roots:
        if not root.exists():
            continue
        for outcomes_path in sorted(root.rglob("outcomes.jsonl")):
            config = _read_yaml(outcomes_path.parent / "config_snapshot.yaml")
            inferred = config.get("suite") or config.get("suite_name") or (config.get("metadata") or {}).get("suite_name")
            if inferred != suite and suite not in outcomes_path.parts and not outcomes_path.parent.name.startswith(f"{suite}__"):
                continue
            rows.extend(read_jsonl(outcomes_path))
    return rows


def _run_has_official_bigcodebench_labels(run: Any) -> bool:
    return bool(run.outcomes) and all(
        (row.get("metadata") or {}).get("label_source") == "official_bigcodebench" for row in run.outcomes
    )


def _replacement_config(suite: str) -> dict[str, Any]:
    candidates = [
        Path(f"configs/{suite}_provider.yaml"),
        Path(f"configs/{suite}.yaml"),
    ]
    for path in candidates:
        config = _read_yaml(path)
        if config:
            return config
    return {}


def _config_models(config: dict[str, Any]) -> list[str]:
    if config.get("model"):
        return [str(config["model"])]
    models: list[str] = []
    for entry in config.get("models") or []:
        if isinstance(entry, dict):
            value = entry.get("name") or entry.get("model")
        else:
            value = entry
        if value:
            models.append(str(value))
    return [model for model in models if model != "mock-model"] or [""]


def _bigcodebench_official_roots(roots: Iterable[str | Path] | None) -> list[Path]:
    selected = [Path(root) for root in roots] if roots is not None else []
    direct = Path("reports/artifacts_bigcodebench_official/paper_bigcodebench_hard")
    parent = Path("reports/artifacts_bigcodebench_official")
    default = direct if direct.exists() else parent
    if default.exists() and default not in selected:
        selected.append(default)
    return selected


def _language_by_task(task_file: Any) -> dict[str, str]:
    if not task_file or not Path(str(task_file)).exists():
        return {}
    result: dict[str, str] = {}
    for row in read_jsonl(Path(str(task_file))):
        task = TaskRecord.model_validate(row)
        metadata = task.metadata or {}
        external = task.external_eval or {}
        result[task.task_id] = str(metadata.get("language") or external.get("language") or "unknown")
    return result


def _pass_rate_by_budget_text(outcomes: list[dict[str, Any]]) -> str:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in outcomes:
        if row.get("budget") is not None and "success" in row:
            grouped[int(row["budget"])].append(bool(row["success"]))
    return json.dumps({str(budget): sum(values) / len(values) for budget, values in sorted(grouped.items())}, sort_keys=True)


def _budget_grid_text(outcomes: list[dict[str, Any]], *, fallback: str = "") -> str:
    budgets = sorted({int(row["budget"]) for row in outcomes if row.get("budget") is not None})
    return ",".join(str(budget) for budget in budgets) if budgets else fallback


def _metric_text(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _agentic_bridge_rows() -> list[dict[str, Any]]:
    estimate = _read_json(Path("reports/tables/paper_agentic_swe_bridge_cost_estimate.json"))
    local_tasks = _jsonl_count(Path("data/processed/swe_verified_smoke.jsonl"))
    run_root = Path("reports/runs/paper_agentic_swe_bridge")
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    if run_dirs:
        rows: list[dict[str, Any]] = []
        for run_dir in sorted(run_dirs):
            outcomes_path = run_dir / "outcomes.jsonl"
            forecasts_path = run_dir / "forecasts.jsonl"
            if not outcomes_path.exists():
                continue
            outcomes = read_jsonl(outcomes_path)
            forecasts = read_jsonl(forecasts_path) if forecasts_path.exists() else []
            budgets = sorted({int(row.get("budget")) for row in outcomes if row.get("budget") is not None})
            success_by_budget = {
                f"success_at_{budget}": (
                    sum(1 for row in outcomes if row.get("budget") is not None and int(row.get("budget")) == budget and row.get("success")) /
                    max(1, sum(1 for row in outcomes if row.get("budget") is not None and int(row.get("budget")) == budget))
                )
                for budget in budgets
            }
            rows.append(
                {
                    "suite": "paper_agentic_swe_bridge",
                    "local_tasks_available": local_tasks,
                    "model": _first_value(outcomes, "model") or _first_value(forecasts, "model") or run_dir.name,
                    "models": _first_value(outcomes, "model") or _first_value(forecasts, "model") or run_dir.name,
                    "budget_grid": ",".join(str(budget) for budget in budgets) if budgets else "4096,16384",
                    "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
                    "cap_usd": estimate.get("cap_usd", "") if estimate else 20,
                    "run_status": "completed_live_api",
                    "budget_control": "generated_token_cap_pilot",
                    "paper_role": "appendix_bridge_only",
                    "official_harness_status": "not_run",
                    **success_by_budget,
                    "notes": "Generated-token-cap SWE bridge pilot; SWE-bench official harness labels are not claimed.",
                }
            )
        if rows:
            return rows
    return [
        {
            "suite": "paper_agentic_swe_bridge",
            "local_tasks_available": local_tasks,
            "models": ",".join(estimate.get("models", [])) if estimate else "gemini-2.0-flash-lite-001",
            "budget_grid": "4096,16384",
            "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
            "cap_usd": estimate.get("cap_usd", "") if estimate else 20,
            "run_status": "skipped_missing_BUDGET2SUCCESS_GATEWAY_API_KEY",
            "budget_control": "generated_token_cap_pilot",
            "paper_role": "appendix_bridge_only",
            "notes": "Not part of the main claim; full agentic total-token budget control remains future work.",
        }
    ]


def _swe_official_mini_rows(official_artifact_roots: Iterable[str | Path] | None = None) -> list[dict[str, Any]]:
    estimate = _read_json(Path("reports/tables/paper_swe_verified_mini_official_cost_estimate.json"))
    config = _read_yaml(Path("configs/paper_swe_verified_mini_official.yaml"))
    task_file = Path(str(config.get("task_file") or "data/processed/paper_swe_verified_mini_official.jsonl"))
    local_tasks = _jsonl_count(task_file)
    roots = _swe_official_roots(official_artifact_roots)
    runs = []
    for root in roots:
        try:
            runs.extend(
                load_paper_runs(
                    suite="paper_swe_verified_mini_official",
                    run_root="reports/runs",
                    artifact_root=root,
                    include_artifacts=True,
                    official_artifact_roots=[root],
                )
            )
        except FileNotFoundError:
            continue
    rows: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for run in runs:
        if not run.outcomes:
            continue
        if run.model in seen_models:
            continue
        seen_models.add(run.model)
        budgets = sorted({int(row["budget"]) for row in run.outcomes if row.get("budget") is not None})
        success_by_budget = {
            f"success_at_{budget}": (
                sum(1 for row in run.outcomes if int(row.get("budget") or 0) == budget and row.get("success")) /
                max(1, sum(1 for row in run.outcomes if int(row.get("budget") or 0) == budget))
            )
            for budget in budgets
        }
        official_labels = sum(1 for row in run.outcomes if (row.get("metadata") or {}).get("label_source") == "official_swebench")
        error_labels = sum(1 for row in run.outcomes if (row.get("metadata") or {}).get("label_source") == "official_swebench_error")
        status = "completed" if official_labels else ("failed_or_incomplete" if error_labels else "not_merged")
        scored = score_curve_set(
            forecast_curves(run.forecasts),
            outcomes_by_task(run.outcomes),
            predicted_ttg_by_task=forecast_medians(run.forecasts),
            outcome_rows=run.outcomes,
        )
        rows.append(
            {
                "suite": "paper_swe_verified_mini_official",
                "model": run.model,
                "n_tasks": scored.get("n_tasks") or len({row.get("task_id") for row in run.outcomes}),
                "budget_grid": ",".join(str(budget) for budget in budgets),
                "official_harness_status": status,
                "official_labels": official_labels,
                "official_label_errors": error_labels,
                "success_at_max_budget": scored.get("success_at_max_budget", ""),
                "brier": scored.get("brier", ""),
                "ece": scored.get("ece", ""),
                "regret": scored.get("regret", ""),
                "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
                "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 45),
                **success_by_budget,
                "notes": (
                    "Official SWE-bench labels merged."
                    if official_labels
                    else "Official SWE-bench harness failed or labels were unavailable; appendix infrastructure only."
                ),
            }
        )
    if rows:
        return rows
    run_root = Path("reports/runs/paper_swe_verified_mini_official")
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
    status = "raw_runs_without_official_labels" if any((path / "outcomes.jsonl").exists() for path in run_dirs) else "configured_not_run"
    return [
        {
            "suite": "paper_swe_verified_mini_official",
            "model": "",
            "n_tasks": local_tasks,
            "budget_grid": "4096,16384",
            "official_harness_status": status,
            "official_labels": 0,
            "official_label_errors": 0,
            "success_at_max_budget": "",
            "brier": "",
            "ece": "",
            "regret": "",
            "estimated_cost_usd": estimate.get("estimated_total_cost_usd", "") if estimate else "",
            "cap_usd": estimate.get("cap_usd", "") if estimate else config.get("metadata", {}).get("cost_cap_usd", 45),
            "notes": "Appendix infrastructure only: SWE-bench Verified mini bridge is not a main result without official labels.",
        }
    ]


def _placeholder_unverified(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    verification_metadata = verification.get("metadata") if isinstance(verification.get("metadata"), dict) else {}
    details = verification.get("details") if isinstance(verification.get("details"), dict) else {}
    return (
        metadata.get("label_source") == "official_harness_placeholder"
        or verification_metadata.get("label_source") == "official_harness_placeholder"
        or metadata.get("exclude_from_main_metrics") is True
        or verification_metadata.get("exclude_from_main_metrics") is True
        or details.get("error") == "official_harness_required"
    )


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key) not in {"", None}:
            return row.get(key)
    return None


def _agreement_rate(values: list[bool]) -> float:
    if len(values) < 2:
        return 1.0
    pairs = 0
    agree = 0
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            pairs += 1
            agree += 1 if left == right else 0
    return agree / pairs if pairs else 1.0


def _repeatability_nonmonotonicity(
    by_task_repeat: dict[tuple[str, str, str, str, str], dict[int, bool]]
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for (suite, model, source, _task_id, _repeat_id), by_budget in by_task_repeat.items():
        seen_success = False
        nonmonotone = False
        for _budget, success in sorted(by_budget.items()):
            if seen_success and not success:
                nonmonotone = True
                break
            seen_success = seen_success or bool(success)
        grouped[(suite, model, source)].append(nonmonotone)
    return {
        key: sum(1.0 for value in values if value) / len(values)
        for key, values in grouped.items()
        if values
    }


def _verifier_artifact_rows(runs) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for run in runs:
        for task_file in [run.config.get("task_file")]:
            if not task_file or not Path(task_file).exists():
                continue
            for raw in read_jsonl(task_file):
                task = TaskRecord.model_validate(raw)
                entry = by_source.setdefault(
                    task.source,
                    {
                        "source": task.source,
                        "verifier_type": task.verifier,
                        "official_verifier_used": _official_verifier(task.verifier),
                        "version_or_commit": task.source_version or "",
                        "raw_artifacts_present": True,
                        "notes": set(),
                    },
                )
                entry["notes"].add(f"{task.track}; {task.external_eval.get('harness', task.verifier)}")
        if not all((run.run_dir / name).exists() for name in ["forecasts.jsonl", "outcomes.jsonl", "metrics.json"]):
            for entry in by_source.values():
                entry["raw_artifacts_present"] = False
    rows = []
    for entry in sorted(by_source.values(), key=lambda row: row["source"]):
        rows.append({**entry, "notes": "; ".join(sorted(entry["notes"]))})
    return rows


def _artifact_outcome_line_count(run) -> int | None:
    path = run.run_dir / "outcomes.jsonl"
    if path.exists():
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None


def _table6_cost_rows(runs, *, final_paper_mode: bool = False) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        artifact_rows = _artifact_outcome_line_count(run)
        row_count_matches = artifact_rows is None or artifact_rows == len(run.outcomes)
        if final_paper_mode and not row_count_matches:
            raise ValueError(
                f"Cost/runtime row count mismatch for {run.run_dir}: "
                f"loaded={len(run.outcomes)} artifact_rows={artifact_rows}"
            )
        prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in run.outcomes)
        completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in run.outcomes)
        forecast_prompt_tokens = sum(int((row.get("metadata") or {}).get("prompt_tokens") or 0) for row in run.forecasts)
        forecast_completion_tokens = sum(int((row.get("metadata") or {}).get("completion_tokens") or 0) for row in run.forecasts)
        wall_times = [float(row["wall_time_seconds"]) for row in run.outcomes if row.get("wall_time_seconds") is not None]
        rows.append(
            {
                "suite": run.suite or "",
                "run_id": run.run_id,
                "model": run.model,
                "artifact_source": run.artifact_source,
                "budgeted_outcomes": len(run.outcomes),
                "artifact_outcome_rows": artifact_rows if artifact_rows is not None else "",
                "row_count_matches_artifact": row_count_matches,
                "prompt_tokens": prompt_tokens + forecast_prompt_tokens,
                "completion_tokens": completion_tokens + forecast_completion_tokens,
                "mean_wall_time_seconds": sum(wall_times) / len(wall_times) if wall_times else "",
                "estimated_cost_usd": _cost_for_run(run),
                "pricing_config_version": "2026-04-28",
                "cost_mode": "active_artifact_estimate",
                "reasoning_tokens_available": any(row.get("reasoning_tokens") is not None for row in run.outcomes),
                "notes": (
                    "Generated-token budgets are solver output caps; hidden reasoning tokens are reported only when providers expose them."
                    if row_count_matches
                    else f"ROW COUNT MISMATCH: loaded {len(run.outcomes)} outcomes but artifact has {artifact_rows} rows."
                ),
            }
        )
    if rows:
        present_artifact_counts = [
            int(row["artifact_outcome_rows"])
            for row in rows
            if row.get("artifact_outcome_rows") not in {"", None}
        ]
        total_matches_artifacts = all(str(row.get("row_count_matches_artifact")).lower() == "true" for row in rows)
        rows.append(
            {
                "suite": "TOTAL",
                "run_id": "",
                "model": "",
                "artifact_source": "",
                "budgeted_outcomes": sum(int(row.get("budgeted_outcomes") or 0) for row in rows),
                "artifact_outcome_rows": sum(present_artifact_counts) if present_artifact_counts else "",
                "row_count_matches_artifact": total_matches_artifacts,
                "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
                "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
                "mean_wall_time_seconds": "",
                "estimated_cost_usd": round(
                    sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows if row.get("estimated_cost_usd") not in {"", None}),
                    6,
                ),
                "pricing_config_version": "2026-04-28",
                "cost_mode": "sum_of_included_active_artifact_rows",
                "reasoning_tokens_available": any(str(row.get("reasoning_tokens_available")).lower() == "true" for row in rows),
                "notes": (
                    "Total over included main paper artifact rows; provider invoices may differ from token-based estimates."
                    if total_matches_artifacts
                    else "ROW COUNT MISMATCH in at least one included artifact row."
                ),
            }
        )
    return rows


def _official_verifier(verifier: str) -> bool:
    return verifier in {"evalplus", "bigcodebench", "livecodebench", "swebench", "bfcl", "tau2"}


def _metric_definition_rows() -> list[dict[str, str]]:
    return [
        {
            "metric": "brier",
            "family": "calibration",
            "definition": "Mean squared error between forecast P(success by budget) and verified binary outcomes.",
            "direction": "lower is better",
        },
        {
            "metric": "ece",
            "family": "calibration",
            "definition": "Expected calibration error over budget-outcome probability bins.",
            "direction": "lower is better",
        },
        {
            "metric": "success_at_max_budget",
            "family": "capability",
            "definition": "Share of tasks solved at the largest observed budget.",
            "direction": "higher is better",
        },
        {
            "metric": "censoring_rate",
            "family": "success_budget",
            "definition": "Share of tasks not solved by the largest observed budget.",
            "direction": "lower is better",
        },
        {
            "metric": "absolute_log_budget_error",
            "family": "success_budget",
            "definition": "Absolute log error between predicted median success budget and first verified success budget, solved tasks only.",
            "direction": "lower is better",
        },
        {
            "metric": "signed_log_budget_error",
            "family": "allocation",
            "definition": "Mean log(predicted budget) minus log(observed first-success budget); negative means underbudgeting and positive means overbudgeting.",
            "direction": "closer to 0 is better",
        },
        {
            "metric": "censored_lower_bound_error",
            "family": "success_budget",
            "definition": "Lower-bound log error for right-censored tasks when forecasts fall below the censoring budget.",
            "direction": "lower is better",
        },
        {
            "metric": "underbudget_rate",
            "family": "allocation",
            "definition": "Share of solved tasks where predicted budget is below observed first-success budget.",
            "direction": "lower is better",
        },
        {
            "metric": "overbudget_rate",
            "family": "allocation",
            "definition": "Share of solved tasks where predicted budget is above observed first-success budget.",
            "direction": "lower is better",
        },
        {
            "metric": "underbudget_shortfall_factor",
            "family": "allocation",
            "definition": "Observed first-success budget divided by predicted budget on underbudgeted solved tasks.",
            "direction": "lower is better",
        },
        {
            "metric": "overbudget_waste_factor",
            "family": "allocation",
            "definition": "Predicted budget divided by observed first-success budget on overbudgeted solved tasks.",
            "direction": "lower is better",
        },
        {
            "metric": "regret",
            "family": "allocation",
            "definition": "Utility gap between the forecast-selected budget and the best verified budget in hindsight.",
            "direction": "lower is better",
        },
        {
            "metric": "normalized_regret",
            "family": "allocation",
            "definition": "Budget-selection regret divided by the observed oracle utility range; zero-range tasks are assigned 0.0.",
            "direction": "lower is better",
        },
        {
            "metric": "forecast_monotonicity_violation_rate",
            "family": "diagnostic",
            "definition": "Share of task forecast curves that decrease as budget increases.",
            "direction": "lower is better",
        },
        {
            "metric": "outcome_nonmonotonicity_rate",
            "family": "diagnostic",
            "definition": "Share of tasks that pass at one budget and fail at a later larger budget.",
            "direction": "lower is better",
        },
        {
            "metric": "task_budget_ranking_accuracy",
            "family": "diagnostic",
            "definition": "Pairwise accuracy for ranking solved tasks by required budget from predicted success budget.",
            "direction": "higher is better",
        },
        {
            "metric": "truncation_rate",
            "family": "diagnostic",
            "definition": "Share of budgeted solver outputs ending by length or at the generated-token cap.",
            "direction": "report",
        },
    ]


def _release_checklist_rows() -> list[dict[str, str]]:
    checks = [
        ("forecasts_packaged", Path("reports/artifacts").exists(), "Packaged forecast JSONL artifacts"),
        ("outcomes_packaged", any(Path("reports/artifacts").glob("*/outcomes.jsonl")), "Packaged outcome JSONL artifacts"),
        ("manifest_present", any(Path("reports/artifacts").glob("*/sha256_manifest.json")), "Per-run SHA-256 manifests"),
        ("readme_present", Path("README.md").exists(), "Public README"),
        ("benchmark_card", Path("BENCHMARK_CARD.md").exists(), "Benchmark card"),
        ("data_provenance", Path("DATA_PROVENANCE.md").exists(), "Data provenance note"),
        ("reproducing_guide", Path("REPRODUCING.md").exists(), "Reproduction guide"),
        ("croissant_metadata", Path("metadata/croissant.json").exists(), "Croissant metadata"),
        ("release_manifest", Path("reports/release_manifest.json").exists(), "Release manifest"),
        ("leakage_audit", Path("reports/tables/forecast_leakage_audit.csv").exists(), "Forecast leakage audit"),
        ("calibration_eval_splits", Path("reports/splits").exists() and any(Path("reports/splits").glob("*_calibration_eval_split.json")), "Calibration/evaluation split files"),
        ("math_reverification", Path("reports/tables/math_verifier_delta_summary.csv").exists(), "Strict math verifier delta summary"),
        ("math_answer_type_audit", Path("reports/tables/paper_table_math_answer_type_audit.csv").exists(), "Math answer-type audit table"),
        ("repeatability_audit", Path("reports/tables/paper_table10_repeatability.csv").exists(), "Repeatability audit table"),
        ("secret_scrub_audit", Path("reports/tables/secret_scrub_audit.csv").exists(), "Secret scrub audit"),
        ("submission_validation", Path("reports/tables/submission_package_validation.csv").exists(), "Submission package validation audit"),
    ]
    return [
        {"check": name, "status": "present" if ok else "missing", "notes": notes}
        for name, ok, notes in checks
    ]


def _ci(metrics: dict[str, dict[str, str]], key: str) -> str:
    row = metrics.get(key)
    if not row:
        return ""
    rendered = ci_string(row.get("estimate"), row.get("ci_low"), row.get("ci_high"))
    if rendered:
        return rendered
    estimate = row.get("estimate")
    return _point_ci(estimate)


def _point_ci(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f} [point]"
    except Exception:
        return str(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _live_costs() -> dict[str, float]:
    path = Path("reports/live_runs/provider_live_summary.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["run_id"]: float(row.get("estimated_cost_usd") or 0.0) for row in csv.DictReader(f)}


def _cost_for_run(run) -> float | str:
    return _live_costs().get(run.run_id, _cost_from_usage(run.model, run.forecasts, run.outcomes))


def _cost_from_usage(model: str, forecasts: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> float | str:
    pricing_path = Path("reports/live_runs/provider_live_cost_estimate.json")
    if not pricing_path.exists():
        return ""
    pricing = json.loads(pricing_path.read_text(encoding="utf-8")).get("pricing", {})
    rates = pricing.get(model)
    if not rates:
        return ""
    input_tokens = sum(int((row.get("metadata") or {}).get("prompt_tokens") or 0) for row in forecasts)
    output_tokens = sum(int((row.get("metadata") or {}).get("completion_tokens") or 0) for row in forecasts)
    input_tokens += sum(int(row.get("prompt_tokens") or 0) for row in outcomes)
    output_tokens += sum(int(row.get("completion_tokens") or 0) for row in outcomes)
    return round(
        input_tokens / 1_000_000.0 * float(rates["input_per_m"])
        + output_tokens / 1_000_000.0 * float(rates["output_per_m"]),
        6,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    from budget2success.utils.config import load_yaml

    return load_yaml(path)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _official_roots(roots: Iterable[str | Path] | None) -> list[Path]:
    selected = [Path(root) for root in roots] if roots is not None else []
    default = Path("reports/artifacts_livecodebench_official")
    if default.exists() and default not in selected:
        selected.append(default)
    return selected


def _swe_official_roots(roots: Iterable[str | Path] | None) -> list[Path]:
    selected = [Path(root) for root in roots] if roots is not None else []
    direct = Path("reports/artifacts_swebench_official/paper_swe_verified_mini_official")
    parent = Path("reports/artifacts_swebench_official")
    default = direct if direct.exists() else parent
    if default.exists() and default not in selected:
        selected.append(default)
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _assert_no_forbidden_main_table_rows(paths: Iterable[Path]) -> None:
    violations: list[str] = []
    for path in paths:
        if not path.name.startswith("paper_table") or path.suffix != ".csv":
            continue
        rows = _read_csv(path)
        for index, row in enumerate(rows, start=2):
            text = json.dumps(row, sort_keys=True).lower()
            if "mock-model" in text:
                violations.append(f"{path}:{index}: mock-model")
                continue
            role = str(row.get("paper_role") or "").strip().lower()
            main_like = role in {"main", "paper", "main_text", "main_candidate"}
            if main_like and any(value in text for value in FORBIDDEN_MAIN_TABLE_VALUES - {"mock-model"}):
                violations.append(f"{path}:{index}: forbidden main evidence row")
    if violations:
        preview = "; ".join(violations[:5])
        raise AssertionError(f"Forbidden paper evidence rows detected: {preview}")


def _write_artifact_checklist_doc(table_path: Path) -> None:
    rows = _read_csv(table_path)
    lines = [
        "# Paper Artifact Checklist",
        "",
        "Generated from `scripts/make_paper_tables.py`.",
        "",
        f"- Source table: `{table_path}`",
        f"- SHA-256: `{sha256_file(table_path) if table_path.exists() else ''}`",
        "",
    ]
    if rows:
        lines.extend(_markdown_table(rows).splitlines())
    path = Path("reports/tables/paper_artifact_checklist.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables from frozen TokenCapBench outputs.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--table-dir", default=str(TABLE_DIR))
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--corrected-artifact-root", default=None)
    parser.add_argument("--official-artifact-root", action="append", default=None)
    parser.add_argument(
        "--math-label-mode",
        choices=["original", "strict", "corrected"],
        default="original",
        help="'strict' is retained as an alias for task-default corrected math labels.",
    )
    parser.add_argument("--final-paper-mode", action="store_true")
    args = parser.parse_args()
    outputs = make_paper_tables(
        suite=args.suite,
        table_dir=args.table_dir,
        artifact_root=args.artifact_root,
        include_artifacts=not args.no_artifacts,
        corrected_artifact_root=args.corrected_artifact_root,
        official_artifact_roots=args.official_artifact_root,
        math_label_mode=args.math_label_mode,
        final_paper_mode=args.final_paper_mode,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
