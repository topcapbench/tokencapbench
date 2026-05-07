#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.utils.manifest import git_commit, sha256_file


OUTPUTS = [
    "reports/tables/bootstrap_main_metrics.csv",
    "reports/tables/bootstrap_success_by_budget.csv",
    "reports/tables/calibration_eval_split_summary.csv",
    "reports/tables/baseline_comparison.csv",
    "reports/tables/baseline_summary.csv",
    "reports/tables/math_reverification_audit.csv",
    "reports/tables/math_reverification_corrections.jsonl",
    "reports/tables/math_verifier_delta_summary.csv",
    "reports/tables/paper_table_math_answer_type_audit.csv",
    "reports/tables/paper_table_repeatability_audit.csv",
    "reports/tables/paper_table10_repeatability.csv",
    "reports/tables/paper_table11_fresh_coding.csv",
    "reports/tables/paper_table3a_calibration_capability.csv",
    "reports/tables/paper_table3b_allocation_diagnostics.csv",
    "reports/tables/paper_table4_calibration_split_baselines.csv",
    "reports/tables/paper_table4_main_baseline_summary.csv",
    "reports/tables/paper_table5_diagnostic_baselines.csv",
    "reports/tables/paper_table7_verifier_robustness.csv",
    "reports/tables/paper_table_manual_math_audit_summary.csv",
    "reports/tables/paper_table8_metric_definitions.csv",
    "reports/tables/submission_package_validation.csv",
    "reports/tables/secret_scrub_audit.csv",
    "reports/figures/paper_figure3_calibration_by_suite.png",
    "reports/figures/paper_figure4_budget_error_distribution.png",
    "reports/figures/paper_figure5_calibration_split_baselines.png",
    "reports/figures/paper_figure8_diagnostics.png",
    "reports/figures/appendix_repeatability_audit.png",
    "reports/figures/appendix_repeatability_variance.png",
    "reports/figures/appendix_normalized_regret.png",
    "reports/figures/appendix_fresh_coding.png",
    "reports/paper_results_report.md",
    "metadata/croissant.json",
]


def reproduce_paper_artifacts(
    *,
    artifact_root: str | Path = "reports/artifacts",
    corrected_artifact_root: str | Path = "reports/artifacts_corrected",
    split_dir: str | Path = "reports/splits",
    math_label_mode: str = "original",
    final_paper_mode: bool = False,
    cap_usd: float = 40.0,
    n_bootstrap: int = 250,
    ranking_max_pairs: int = 10000,
    live_api_calls_made: bool = False,
    new_api_spend_usd: float = 0.0,
    extra_commands: list[str] | None = None,
) -> Path:
    artifact_root = Path(artifact_root)
    corrected_artifact_root = Path(corrected_artifact_root)
    split_dir = Path(split_dir)
    bootstrap_command = [
        sys.executable,
        "scripts/bootstrap_metrics.py",
        "--artifact-root",
        str(artifact_root),
        "--n-bootstrap",
        str(n_bootstrap),
        "--ranking-max-pairs",
        str(ranking_max_pairs),
    ]
    if math_label_mode in {"strict", "corrected"}:
        bootstrap_command.extend(
            ["--corrected-artifact-root", str(corrected_artifact_root), "--math-label-mode", "corrected"]
        )
    tables_command = [sys.executable, "scripts/make_paper_tables.py", "--artifact-root", str(artifact_root)]
    figures_command = [sys.executable, "scripts/make_paper_figures.py", "--artifact-root", str(artifact_root)]
    official_livecodebench_root = Path("reports/artifacts_livecodebench_official")
    if official_livecodebench_root.exists():
        tables_command.extend(["--official-artifact-root", str(official_livecodebench_root)])
    if final_paper_mode:
        tables_command.append("--final-paper-mode")
    if math_label_mode in {"strict", "corrected"}:
        tables_command.extend(["--corrected-artifact-root", str(corrected_artifact_root), "--math-label-mode", "corrected"])
        figures_command.extend(["--corrected-artifact-root", str(corrected_artifact_root), "--math-label-mode", "corrected"])
    commands = [
        [
            sys.executable,
            "scripts/build_calibration_eval_splits.py",
            "--artifact-root",
            str(artifact_root),
            "--output-dir",
            str(split_dir),
            "--calibration-frac",
            "0.30",
            "--seed",
            "20260428",
        ],
        [
            sys.executable,
            "scripts/reverify_outcomes.py",
            "--artifact-root",
            str(artifact_root),
            "--mode",
            "task_aware_strict",
            "--output",
            "reports/tables/math_reverification_audit.csv",
            "--write-corrections",
            "reports/tables/math_reverification_corrections.jsonl",
            "--write-corrected-outcomes",
            str(corrected_artifact_root),
        ],
        bootstrap_command,
        [
            sys.executable,
            "scripts/run_baseline_analysis.py",
            "--artifact-root",
            str(artifact_root),
            "--split-dir",
            str(split_dir),
            "--use-calibration-split",
            "--bootstrap",
            "--include-test-distribution-diagnostics",
            "--n-bootstrap",
            str(n_bootstrap),
            "--seed",
            "20260428",
        ],
        [sys.executable, "scripts/audit_forecast_leakage.py"],
        tables_command,
        figures_command,
        [
            sys.executable,
            "scripts/make_paper_results_report.py",
            "--artifact-root",
            str(artifact_root),
            "--corrected-artifact-root",
            str(corrected_artifact_root),
            "--math-label-mode",
            "corrected" if math_label_mode in {"strict", "corrected"} else "original",
        ],
        [sys.executable, "scripts/make_croissant_metadata.py", "--artifact-root", str(artifact_root)],
        [sys.executable, "scripts/scrub_artifacts_for_release.py"],
        [
            sys.executable,
            "scripts/validate_submission_package.py",
            "--artifact-root",
            str(artifact_root),
            "--tables-dir",
            "reports/tables",
            "--figures-dir",
            "reports/figures",
            "--metadata-dir",
            "metadata",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    return _write_release_manifest(
        artifact_root=artifact_root,
        cap_usd=cap_usd,
        commands=commands,
        corrected_artifact_root=corrected_artifact_root,
        split_dir=split_dir,
        final_paper_mode=final_paper_mode,
        math_label_mode=math_label_mode,
        live_api_calls_made=live_api_calls_made,
        new_api_spend_usd=new_api_spend_usd,
        extra_commands=extra_commands or [],
    )


def _write_release_manifest(
    *,
    artifact_root: Path,
    cap_usd: float,
    commands: list[list[str]],
    corrected_artifact_root: Path,
    split_dir: Path,
    final_paper_mode: bool,
    math_label_mode: str,
    live_api_calls_made: bool,
    new_api_spend_usd: float,
    extra_commands: list[str],
) -> Path:
    rendered_math_label_mode = "corrected" if math_label_mode in {"strict", "corrected"} else math_label_mode
    files: dict[str, str] = {}
    file_candidates = set(OUTPUTS)
    for root in [artifact_root, Path("reports/tables"), Path("reports/figures"), Path("metadata"), split_dir, corrected_artifact_root]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".pdf", ".png", ".svg"}:
                    file_candidates.add(str(path))
    for value in sorted(file_candidates):
        path = Path(value)
        if path.exists():
            files[value] = sha256_file(path)
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "git_dirty": _git_dirty(),
        "artifact_root": str(artifact_root),
        "corrected_artifact_root": str(corrected_artifact_root),
        "split_dir": str(split_dir),
        "final_paper_mode": final_paper_mode,
        "math_label_mode": rendered_math_label_mode,
        "budget_cap_usd": cap_usd,
        "live_api_calls_made": bool(live_api_calls_made),
        "new_api_spend_usd": float(new_api_spend_usd),
        "commands": [" ".join(command) for command in commands] + list(extra_commands),
        "files": files,
    }
    output = Path("reports/release_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--short"], text=True, capture_output=True, check=True)
    except Exception:
        return None
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper tables, figures, report, and release metadata from frozen artifacts.")
    parser.add_argument("--artifact-root", "--from-artifacts", dest="artifact_root", default="reports/artifacts")
    parser.add_argument("--corrected-artifact-root", default="reports/artifacts_corrected")
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument(
        "--math-label-mode",
        choices=["original", "strict", "corrected"],
        default="original",
        help="'strict' is retained as an alias for task-default corrected math labels.",
    )
    parser.add_argument("--final-paper-mode", action="store_true")
    parser.add_argument("--cap-usd", type=float, default=40.0)
    parser.add_argument("--n-bootstrap", type=int, default=250)
    parser.add_argument("--ranking-max-pairs", type=int, default=10000)
    parser.add_argument("--live-api-calls-made", action="store_true")
    parser.add_argument("--new-api-spend-usd", type=float, default=0.0)
    parser.add_argument("--extra-command", action="append", default=[])
    args = parser.parse_args()
    path = reproduce_paper_artifacts(
        artifact_root=args.artifact_root,
        corrected_artifact_root=args.corrected_artifact_root,
        split_dir=args.split_dir,
        math_label_mode=args.math_label_mode,
        final_paper_mode=args.final_paper_mode,
        cap_usd=args.cap_usd,
        n_bootstrap=args.n_bootstrap,
        ranking_max_pairs=args.ranking_max_pairs,
        live_api_calls_made=args.live_api_calls_made,
        new_api_spend_usd=args.new_api_spend_usd,
        extra_commands=args.extra_command,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
