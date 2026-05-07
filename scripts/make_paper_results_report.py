#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_paper_runs


REPORT_PATH = Path("reports/paper_results_report.md")
DEFAULT_SUITE_LABEL = "default paper suites (math core, HumanEval+, MBPP+)"


def make_paper_results_report(
    *,
    suite: str | None = None,
    output: str | Path = REPORT_PATH,
    artifact_root: str | Path | None = "reports/artifacts",
    include_artifacts: bool = True,
    corrected_artifact_root: str | Path | None = None,
    math_label_mode: str = "original",
) -> Path:
    load_kwargs = {
        "suite": suite,
        "run_root": Path(artifact_root) / "__no_reports_runs__" if artifact_root is not None else "reports/runs",
        "artifact_root": artifact_root,
        "include_artifacts": include_artifacts,
        "corrected_artifact_root": corrected_artifact_root,
        "math_label_mode": math_label_mode,
    }
    try:
        runs = load_paper_runs(**load_kwargs)
    except FileNotFoundError:
        if math_label_mode not in {"strict", "corrected"}:
            raise
        load_kwargs["math_label_mode"] = "original"
        load_kwargs["corrected_artifact_root"] = None
        runs = load_paper_runs(**load_kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = _build_report(runs, suite)
    output.write_text(text, encoding="utf-8")
    return output


def _build_report(runs, suite: str | None) -> str:
    evidence_rows = _run_inventory_rows(runs)
    table3a = _read_table(Path("reports/tables/paper_table3a_calibration_capability.csv"))
    table3b = _read_table(Path("reports/tables/paper_table3b_allocation_diagnostics.csv"))
    baseline_rows = _read_table(Path("reports/tables/paper_table4_main_baseline_summary.csv"))
    full_baseline_rows = _read_table(Path("reports/tables/paper_table4_calibration_split_baselines.csv"))
    repeatability_rows = _read_table(Path("reports/tables/paper_table10_repeatability.csv"))
    fresh_rows = _read_table(Path("reports/tables/paper_table11_fresh_coding.csv"))
    fresh_rows = [
        row
        for row in fresh_rows
        if "configured_not_run" not in {row.get("run_status"), row.get("official_harness_status")}
    ]
    bigcodebench_rows = _read_table(Path("reports/tables/paper_table18_bigcodebench_hard.csv"))
    canitedit_rows = _read_table(Path("reports/tables/paper_table19_canitedit_descriptive.csv"))
    cost_runtime_warning = _cost_runtime_warning(Path("reports/tables/paper_table6_cost_runtime.csv"))
    lines = [
        "# TokenCapBench Paper Results Report",
        "",
        "This document is a human-readable paper companion, not a raw artifact dump.",
        "",
        "## Short answer",
        "",
        "**Yes, for the core benchmark.** TokenCapBench now implements the intended forecast-then-execute protocol: a model forecasts success probabilities at generated-token budgets, fresh solver contexts are run under hard caps, deterministic verifiers label the outputs, and the benchmark scores calibration, first-success-budget error, allocation diagnostics, and regret.",
        "",
        "**No, not fully for a final NeurIPS submission yet.** The frozen artifact supports a first paper draft over math plus standalone coding, with chat-completion-compatible BigCodeBench-Hard and CanItEdit extensions reported only when verifier-backed labels are available. SWE, BFCL, OpenHands-style runtimes, and Aider-specific workflows remain future work or infrastructure, not main claims. Final submission still requires hosted distribution URLs for the release/Croissant metadata.",
        "",
        "## Core motivation",
        "",
        "Standard benchmarks ask whether a model can solve a task. TokenCapBench asks a different operational question: before spending tokens, does the model know how much generated-token budget it needs to reach verified success?",
        "",
        "TokenCapBench asks models to forecast `P(verified success by budget B)`. The same model is then run in separate solver contexts under those budget caps. The solver never sees the forecast, so the forecast is evaluated as a pre-execution resource estimate rather than a length-control instruction.",
        "",
        "## Evidence scope",
        "",
        "TokenCapBench is **not** raw output-length prediction. The target is not how many tokens the response will contain; the target is whether verified success is likely by budget `B`.",
        "",
        "TokenCapBench is also **not** a full SWE-agent benchmark in this artifact. SWE-bench, BFCL, OpenHands-style runtimes, Aider-specific workflows, and Docker-heavy repository repair tracks are future work unless official chat-completion-compatible runs are completed.",
        "",
        "The main claim scope is **verified success under hard generated-token budgets** on math plus standalone coding: GSM8K/MATH and EvalPlus HumanEval+/MBPP+. BigCodeBench-Hard is the primary chat-completion-compatible harder code-generation extension, and CanItEdit descriptive is a code-editing bridge when provided-test verification is present. LiveCodeBench freshness evidence uses official labels for completed runs but is not used to claim a full fresh-coding main track unless the paper is updated accordingly.",
        "",
        f"Suite: `{suite or DEFAULT_SUITE_LABEL}`",
        "",
        _compact_metric_table(evidence_rows, max_rows=16) or "No frozen runs found.",
        "",
        "# Results section guide",
        "",
        _figure_section(
            "Figure 1: TokenCapBench protocol",
            "reports/figures/paper_figure1_pipeline.png",
            "The reader needs to see that the protocol is forecast -> fresh solver -> verify -> score.",
            "The model first forecasts a budget-success curve. The solver run is separate and budget-capped by the harness/API.",
            "The unit of evaluation is a model-scaffold-task-budget tuple, which is the right abstraction for budget allocation.",
        ),
        "",
        _table_section(
            "Table 1: Related work matrix",
            _table_or_note("reports/tables/paper_table1_related_work.csv", "Run `scripts/make_paper_tables.py` first."),
            "Reviewers need to distinguish this from time estimation, response-length prediction, token-efficiency, and ordinary task benchmarks.",
            "The table frames TokenCapBench as pre-execution, resource-conditioned success forecasting.",
            "The contribution is the reusable protocol, not the generic fact that tokens matter.",
        ),
        "",
        _table_section(
            "Table 2: Clean evidence scope",
            _table_or_note("reports/tables/paper_table_clean_evidence_scope.csv", "Run `scripts/make_paper_tables.py` first."),
            "A benchmark paper must show exactly which task sources, verifiers, and paper roles support the main claims.",
            "The paper-facing table excludes smoke rows, duplicate raw rows, placeholder labels, and incompatible agent-runtime substrates.",
            "The detailed raw dataset-composition table remains available at `reports/tables/paper_table2_dataset_composition.csv` for artifact provenance.",
        ),
        "",
        _figure_section(
            "Figure 2: Verified success by generated-token budget",
            "reports/figures/paper_figure2_success_by_budget.png",
            "This figure tests whether generated-token budget is a meaningful control variable.",
            "Success generally changes as the token cap changes, and the shape differs by suite and model.",
            "Model capability at the largest budget is not enough; budgeted success curves are operationally relevant.",
        ),
        "",
        _figure_section(
            "Figure 3: Calibration by suite and source",
            "reports/figures/paper_figure3_calibration_by_suite.png",
            "A model can be a strong solver but still give unreliable budget probabilities.",
            "Reliability curves compare predicted success probabilities with observed verified success.",
            "Self-budget forecasting is a separate capability from solving.",
        ),
        "",
        _table_section(
            "Table 3a: Calibration and capability metrics",
            _compact_metric_table(table3a, max_rows=12),
            "The paper needs one compact table for forecast calibration and largest-budget capability.",
            "Brier/ECE, success at max budget, ranking accuracy, and cost are reported together without mixing allocation diagnostics.",
            "Capability and calibration are related but not identical.",
        ),
        "",
        _figure_section(
            "Figure 4: Signed budget-error distribution",
            "reports/figures/paper_figure4_budget_error_distribution.png",
            "Calibration alone does not show whether models tend to request too little or too much budget.",
            "Negative signed log error means underbudgeting; positive values mean overbudgeting.",
            "The benchmark exposes deployment-relevant failure modes, not just aggregate accuracy.",
        ),
        "",
        _table_section(
            "Table 3b: Allocation diagnostics",
            _compact_metric_table(table3b, max_rows=12),
            "Agent designers need underbudgeting, overbudgeting, truncation, regret, and normalized regret in operational terms.",
            "Raw regret is complemented by normalized regret so suites with different budget grids are easier to compare.",
            "Some models waste tokens, some underbudget, and low regret can coexist with imperfect calibration.",
        ),
        "",
        _figure_section(
            "Figure 5: Calibration-split baseline comparison",
            "reports/figures/paper_figure5_calibration_split_baselines.png",
            "Raw self-forecasts must be compared with deployable empirical priors and recalibrators.",
            "The main figure keeps post-hoc output-length diagnostics out of the deployable baseline comparison.",
            "The benchmark is useful because simple calibration-split baselines can beat raw forecasts.",
        ),
        "",
        _table_section(
            "Table 4: Main baseline summary",
            _compact_metric_table(baseline_rows, max_rows=16),
            "The main paper needs a compact comparison of raw forecasts, recalibration, simple priors, and best regret.",
            "This table is one row per suite-model. Full calibration-split details remain in `reports/tables/paper_table4_calibration_split_baselines.csv`.",
            "The paper should not claim current raw LLM forecasts are deployment-ready without calibration.",
        ),
        "",
        _table_section(
            "Appendix Table: Full calibration-split baselines",
            _compact_metric_table(full_baseline_rows, max_rows=8),
            "The compressed main table hides method-level details that reviewers may need.",
            "The full CSV keeps every deployable baseline row and confidence interval.",
            "Main-text brevity should not remove reproducibility detail.",
        ),
        "",
        _figure_section(
            "Figure 6: Budget-selection regret",
            "reports/figures/paper_figure6_regret.png",
            "Regret turns calibrated probabilities into a deployment decision metric.",
            "It compares the utility of the forecast-selected budget against the best observed budget in hindsight.",
            "Forecasts should be evaluated by both calibration and decision usefulness.",
        ),
        "",
        _figure_section(
            "Appendix Figure: Normalized regret",
            "reports/figures/appendix_normalized_regret.png",
            "Normalized regret helps compare deployment loss across suites with different budget grids.",
            "The values are scaled by the observed oracle utility range, with zero-range tasks handled explicitly.",
            "This strengthens the cross-suite deployment argument without crowding Figure 6.",
        ),
        "",
        _table_section(
            "Table 7: Task-aware math verification audit",
            _table_or_note("reports/tables/paper_table7_verifier_robustness.csv", "Run task-aware reverification first."),
            "Math verification is a label-quality risk, especially for symbolic Hendrycks/MATH-style answers.",
            "GSM8K routes to strict numeric verification; MATH-style rows require math-verify in final-paper mode or are marked unsupported.",
            "The final paper should describe verifier policy and unsupported rows explicitly.",
        ),
        "",
        _figure_section(
            "Appendix Figure: Diagnostic baselines",
            "reports/figures/appendix_diagnostic_baselines.png",
            "Post-hoc baselines explain the data but are not deployable pre-execution controllers.",
            "Output-length proxy baselines are separated from the main baseline figure.",
            "Diagnostic baselines should not be used to overstate deployable budget forecasting.",
        ),
        "",
        _table_section(
            "Table 6: Cost and runtime accounting",
            _table_or_note("reports/tables/paper_table6_cost_runtime.csv", "Run `scripts/make_paper_tables.py` first."),
            "A benchmark about token budgets needs transparent token and cost accounting.",
            "The table records forecast and solver token usage, budgeted outcomes, estimated API cost, and reasoning-token availability.",
            (cost_runtime_warning + "\n\n" if cost_runtime_warning else "")
            + "The current artifact is cheap enough to refresh, but provider invoices may differ from token-based estimates.",
        ),
        "",
        _table_section(
            "Table 10: API repeatability",
            _compact_metric_table(repeatability_rows, max_rows=12),
            "API benchmarks should expose whether repeated calls are stable enough for paper claims.",
            "The current live repeatability runs are completed under explicit cost caps; older repeatability rows remain appendix context.",
            "Repeatability strengthens reproducibility but does not replace the main math/coding evidence.",
        ),
        "",
        _table_section(
            "Table 11: Fresh coding",
            _compact_metric_table(fresh_rows, max_rows=6),
            "A fresh coding split addresses contamination and generalization concerns.",
            "The LiveCodeBench split now has official post-hoc labels for the completed fresh-coding runs.",
            "Fresh-coding claims should cite the official LiveCodeBench labels and keep placeholder local labels out of main metrics.",
        ),
        "",
        _table_section(
            "Table 18: BigCodeBench-Hard",
            _compact_metric_table(bigcodebench_rows, max_rows=8),
            "The harder code-generation extension must stay chat-completion compatible.",
            "Rows report official-label status so incomplete BigCodeBench labels cannot become main claims silently.",
            "This is the primary harder coding extension when official BigCodeBench verification is available.",
        ),
        "",
        _table_section(
            "Table 19: CanItEdit descriptive",
            _compact_metric_table(canitedit_rows, max_rows=8),
            "The code-editing bridge uses original code plus a natural-language edit instruction.",
            "Provided-test verification is run locally without Docker; missing hidden tests keep the track appendix-scoped.",
            "This shows how TokenCapBench applies to code editing without relying on an agent runtime.",
        ),
        "",
        "## Final evaluation of results",
        "",
        "### What the current results support",
        "",
        "The current evidence supports the core TokenCapBench protocol, the claim that generated-token budget matters, the claim that self-budget forecasting is not solved, and the practical point that recalibration or simple priors often improve raw self-forecasts. The chat-completion-compatible extension path is BigCodeBench-Hard plus CanItEdit, with labels gated by verifier status.",
        "",
        "### What the current results do not yet support",
        "",
        "The artifact does not yet support full SWE or general agentic claims, model-independent task difficulty claims, complete closed-model hidden-reasoning accounting, or weakly verified fresh-coding claims.",
        "",
        "## Are we ready for a first draft?",
        "",
        "**Yes, we are ready for a first internal paper draft.** The core story, benchmark definition, main figures, main tables, and artifact pipeline are present.",
        "",
        "**No, we are not ready for final NeurIPS submission without caveats.** The final submission still needs hosted release/Croissant URLs, and any optional expansion of LiveCodeBench freshness to all four main models should be clearly scoped if it is not completed.",
        "",
        "## What remains before final submission?",
        "",
        "The remaining submission blocker is final hosted artifact URLs for strict release validation. Optional API extensions can strengthen the appendix, but the main paper scope should remain math plus standalone coding unless the new evidence is completed and reflected in the draft.",
        "",
        "## Reproducibility appendix",
        "",
        "Raw artifact inventory is intentionally kept out of the main narrative. Forecasts, outcomes, metrics, redacted config snapshots, run manifests, and SHA-256 manifests are packaged under `reports/artifacts/`; corrected math labels are under `reports/artifacts_corrected/`.",
        "",
        "```bash",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q",
        "python scripts/reverify_outcomes.py --artifact-root reports/artifacts --mode task_aware_strict --write-corrections reports/tables/math_reverification_corrections.jsonl --write-corrected-outcomes reports/artifacts_corrected",
        "python scripts/reproduce_paper_artifacts.py --artifact-root reports/artifacts --corrected-artifact-root reports/artifacts_corrected --split-dir reports/splits --math-label-mode strict --final-paper-mode --n-bootstrap 1000 --ranking-max-pairs 10000",
        "python scripts/validate_submission_package.py --artifact-root reports/artifacts --tables-dir reports/tables --figures-dir reports/figures --metadata-dir metadata --run-pytest",
        "```",
        "",
    ]
    return "\n".join(lines)


def _run_inventory_rows(runs) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "suite": run.suite or "",
                "run_id": run.run_id,
                "model": run.model,
                "run_dir": str(run.run_dir),
                "artifact_source": run.artifact_source,
                "forecasts": sum(1 for row in run.forecasts if "p_success_by_budget" in row),
                "outcomes": len(run.outcomes),
                "metrics": "yes" if run.metrics else "no",
            }
        )
    return rows


def _read_table(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _compact_metric_table(rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    if not rows:
        return ""
    preferred = [
        "suite",
        "source",
        "run_id",
        "model",
        "forecast_method",
        "raw_self_forecast_brier_ci",
        "best_recalibrated_method",
        "best_recalibrated_brier_ci",
        "best_simple_prior_method",
        "best_simple_prior_brier_ci",
        "best_regret_method",
        "best_regret_ci",
        "n_tasks",
        "n_eval_tasks",
        "forecasts",
        "outcomes",
        "metrics",
        "brier_ci",
        "ece_ci",
        "success_at_max_budget_ci",
        "ranking_accuracy_ci",
        "underbudget_rate_ci",
        "overbudget_rate_ci",
        "regret_ci",
        "normalized_regret_ci",
        "success_agreement_rate",
        "official_harness_status",
        "notes",
    ]
    headers = [header for header in preferred if header in rows[0]]
    if not headers:
        headers = list(rows[0])[:8]
    compact = [{header: row.get(header, "") for header in headers} for row in rows[:max_rows]]
    return _markdown_table(compact)


def _figure_section(title: str, path: str, why: str, interpretation: str, takeaway: str) -> str:
    figure_path = Path(path)
    image = f"![{title}](../{path})" if figure_path.exists() else f"Missing `{path}`"
    return "\n".join(
        [
            f"## {title}",
            "",
            image,
            "",
            "### Why this figure is in the paper",
            "",
            why,
            "",
            "### What it shows in simple language",
            "",
            interpretation,
            "",
            "### Paper takeaway",
            "",
            takeaway,
        ]
    )


def _table_section(title: str, rows_or_summary: str, why: str, interpretation: str, takeaway: str) -> str:
    return "\n".join(
        [
            f"## {title}",
            "",
            "### Why this table is in the paper",
            "",
            why,
            "",
            "### Current summary",
            "",
            rows_or_summary or "No rows available in the current artifact.",
            "",
            "### Interpretation",
            "",
            interpretation,
            "",
            "### Paper takeaway",
            "",
            takeaway,
        ]
    )


def _table_or_note(path: str | Path, note: str) -> str:
    rows = _read_table(Path(path))
    if not rows:
        return note
    return _markdown_table(rows[:20])


def _cost_runtime_warning(path: Path) -> str:
    rows = _read_table(path)
    if not rows:
        return ""
    mismatches = [
        row
        for row in rows
        if row.get("suite") != "TOTAL"
        and str(row.get("row_count_matches_artifact", "")).strip().lower() not in {"true", "1", "yes"}
    ]
    total = next((row for row in rows if row.get("suite") == "TOTAL"), None)
    total_mismatch = bool(
        total
        and str(total.get("row_count_matches_artifact", "")).strip().lower() not in {"true", "1", "yes"}
    )
    if not mismatches and not total_mismatch:
        return ""
    examples = ", ".join(
        f"{row.get('suite')}/{row.get('model')}: loaded={row.get('budgeted_outcomes')} artifact={row.get('artifact_outcome_rows')}"
        for row in mismatches[:4]
    )
    suffix = f" Examples: {examples}." if examples else ""
    return f"**Warning:** cost/runtime row counts do not match artifact outcome counts.{suffix}"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper results report from frozen outputs.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--output", default=str(REPORT_PATH))
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
    path = make_paper_results_report(
        suite=args.suite,
        output=args.output,
        artifact_root=args.artifact_root,
        include_artifacts=not args.no_artifacts,
        corrected_artifact_root=args.corrected_artifact_root,
        math_label_mode=args.math_label_mode,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
