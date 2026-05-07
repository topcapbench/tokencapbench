#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_DOCS = [
    "README.md",
    "BENCHMARK_CARD.md",
    "DATA_PROVENANCE.md",
    "REPRODUCING.md",
]

REQUIRED_TABLES = [
    "paper_table1_related_work.csv",
    "paper_table2_dataset_composition.csv",
    "paper_table_clean_evidence_scope.csv",
    "paper_table3a_calibration_capability.csv",
    "paper_table3b_allocation_diagnostics.csv",
    "paper_table4_calibration_split_baselines.csv",
    "paper_table4_main_baseline_summary.csv",
    "paper_table5_diagnostic_baselines.csv",
    "paper_table6_cost_runtime.csv",
    "paper_table7_verifier_robustness.csv",
    "paper_table8_metric_definitions.csv",
    "paper_table9_release_checklist.csv",
    "paper_table_math_answer_type_audit.csv",
]

REQUIRED_FIGURE_PREFIXES = [
    "paper_figure1_pipeline",
    "paper_figure2_success_by_budget",
    "paper_figure3_calibration_by_suite",
    "paper_figure4_budget_error_distribution",
    "paper_figure5_calibration_split_baselines",
    "paper_figure6_regret",
    "paper_figure7_cost_coverage",
    "paper_figure8_diagnostics",
]

REQUIRED_RELEASE_MANIFEST_FIELDS = [
    "created_utc",
    "code_commit",
    "commands",
    "files",
    "live_api_calls_made",
    "new_api_spend_usd",
    "math_label_mode",
]

REQUIRED_CROISSANT_FIELDS = [
    "@context",
    "@type",
    "name",
    "description",
    "license",
    "version",
    "distribution",
    "recordSet",
]

PLACEHOLDER_DATASET_URL = "https://anonymous.example/tokencapbench"

INCOMPATIBLE_MAIN_TRACK_PATTERNS = [
    "swebench",
    "swe_verified",
    "bfcl",
    "bugsinpy",
    "repoexec",
    "realbench",
    "openhands",
    "aider",
]
STRICT_FIXED_BUDGET_TABLES = [
    "paper_table15_fixed_budget_scheduling.csv",
    "paper_table21_replacement_fixed_budget_scheduling.csv",
]
STRICT_FIXED_BUDGET_COLUMNS = [
    "target_total_budget",
    "selected_total_budget",
    "budget_used",
    "budget_slack_tokens",
    "strict_budget_feasible",
]


def validate_submission_package(
    *,
    artifact_root: str | Path = "reports/artifacts",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    docs_dir: str | Path = "docs",
    metadata_dir: str | Path = "metadata",
    run_pytest: bool = False,
    output: str | Path = "reports/tables/submission_package_validation.csv",
    allow_placeholder_url: bool = False,
    strict_final_submission: bool = False,
) -> tuple[Path, bool]:
    artifact_root = Path(artifact_root)
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    _docs_dir = Path(docs_dir)
    metadata_dir = Path(metadata_dir)
    rows: list[dict[str, Any]] = []

    for doc in REQUIRED_DOCS:
        path = Path(doc)
        _check_path(rows, "required_doc", path, nonempty=True)

    for name in REQUIRED_TABLES:
        _check_path(rows, "required_table", tables_dir / name, nonempty=True)

    for prefix in REQUIRED_FIGURE_PREFIXES:
        for suffix in (".png", ".svg"):
            _check_path(rows, "required_figure", figures_dir / f"{prefix}{suffix}", nonempty=True)

    forecast_files = sorted(artifact_root.glob("*/forecasts.jsonl"))
    outcome_files = sorted(artifact_root.glob("*/outcomes.jsonl"))
    manifests = sorted(artifact_root.glob("*/sha256_manifest.json"))
    _check_condition(rows, "forecast_artifacts", bool(forecast_files), str(artifact_root), f"{len(forecast_files)} forecast JSONL files")
    _check_condition(rows, "outcome_artifacts", bool(outcome_files), str(artifact_root), f"{len(outcome_files)} outcome JSONL files")
    _check_condition(rows, "per_run_manifests", bool(manifests), str(artifact_root), f"{len(manifests)} SHA-256 manifests")

    for path in [metadata_dir / "croissant.json", Path("CITATION.cff"), Path("LICENSE")]:
        _check_path(rows, "metadata", path, nonempty=True)
    _check_release_manifest(rows, Path("reports/release_manifest.json"))
    _check_release_archive(rows, Path("reports/tokencapbench_release_archive.zip"))
    _check_croissant_metadata(
        rows,
        metadata_dir / "croissant.json",
        allow_placeholder_url=allow_placeholder_url,
        strict_final_submission=strict_final_submission,
    )

    _check_path(rows, "secret_scrub", tables_dir / "secret_scrub_audit.csv", nonempty=True)
    _check_secret_scrub(rows, tables_dir / "secret_scrub_audit.csv")
    _check_path(rows, "forecast_leakage_audit", tables_dir / "forecast_leakage_audit.csv", nonempty=True)
    _check_optional_experiment_artifacts(rows, tables_dir, figures_dir)
    _check_main_track_artifacts(rows, artifact_root)
    _check_official_placeholders_not_main(rows, artifact_root)
    _check_incompatible_main_rows(rows)
    _check_fresh_coding_claim(rows, tables_dir)
    _check_replacement_benchmark_outputs(rows, tables_dir, figures_dir)
    _check_clean_evidence_scope(rows, tables_dir / "paper_table_clean_evidence_scope.csv")
    _check_strict_fixed_budget_tables(rows, tables_dir)
    report_path = Path("reports/paper_results_report.md")
    if report_path.exists():
        _check_report_scope(rows, report_path)
    else:
        _check_condition(rows, "paper_report_scope", True, str(report_path), "not present")

    if run_pytest:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            env={**dict(__import__("os").environ), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            text=True,
            capture_output=True,
        )
        _check_condition(
            rows,
            "pytest",
            result.returncode == 0,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q",
            (result.stdout + result.stderr).strip().splitlines()[-1] if (result.stdout + result.stderr).strip() else "",
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    ok = all(row["status"] == "pass" for row in rows)
    return output, ok


def _check_path(rows: list[dict[str, Any]], check: str, path: Path, *, nonempty: bool) -> None:
    exists = path.exists()
    size_ok = exists and (not nonempty or path.stat().st_size > 0)
    rows.append(
        {
            "check": check,
            "status": "pass" if size_ok else "fail",
            "path": str(path),
            "notes": "present" if size_ok else ("missing" if not exists else "empty"),
        }
    )


def _check_condition(rows: list[dict[str, Any]], check: str, ok: bool, path: str, notes: str) -> None:
    rows.append({"check": check, "status": "pass" if ok else "fail", "path": path, "notes": notes})


def _check_release_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        _check_condition(rows, "release_manifest", False, str(path), "missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _check_condition(rows, "release_manifest", False, str(path), f"invalid JSON: {exc}")
        return
    missing = [field for field in REQUIRED_RELEASE_MANIFEST_FIELDS if field not in payload]
    commands_ok = isinstance(payload.get("commands"), list) and bool(payload.get("commands"))
    files_ok = isinstance(payload.get("files"), dict) and bool(payload.get("files"))
    ok = not missing and commands_ok and files_ok
    notes = "present" if ok else f"missing={missing}; commands_ok={commands_ok}; files_ok={files_ok}"
    _check_condition(rows, "release_manifest", ok, str(path), notes)


def _check_release_archive(rows: list[dict[str, Any]], path: Path) -> None:
    _check_path(rows, "release_archive", path, nonempty=True)
    manifest_path = path.with_suffix(path.suffix + ".sha256_manifest.json")
    _check_path(rows, "release_archive_sha256_manifest", manifest_path, nonempty=True)
    if not path.exists() or not manifest_path.exists():
        return
    try:
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except Exception as exc:
        _check_condition(rows, "release_archive_contents", False, str(path), f"invalid zip: {exc}")
        return
    required_suffixes = [
        "README.md",
        "BENCHMARK_CARD.md",
        "DATA_PROVENANCE.md",
        "REPRODUCING.md",
        "forecasts.jsonl",
        "outcomes.jsonl",
        "metrics.json",
        "metadata/croissant.json",
    ]
    missing_suffixes = [suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)]
    required_prefixes = ["reports/tables/", "reports/figures/", "configs/", "prompts/"]
    missing_prefixes = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in names)]
    _check_condition(
        rows,
        "release_archive_contents",
        not missing_suffixes and not missing_prefixes,
        str(path),
        "required release contents present"
        if not missing_suffixes and not missing_prefixes
        else f"missing_suffixes={missing_suffixes}; missing_prefixes={missing_prefixes}",
    )


def _check_secret_scrub(rows: list[dict[str, Any]], path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        _check_condition(rows, "secret_scrub_zero_findings", False, str(path), "missing")
        return
    try:
        with path.open(encoding="utf-8", newline="") as f:
            scrub_rows = list(csv.DictReader(f))
    except Exception as exc:
        _check_condition(rows, "secret_scrub_zero_findings", False, str(path), f"invalid CSV: {exc}")
        return
    finding_rows = [
        row
        for row in scrub_rows
        if str(row.get("kind", "")).strip().lower() not in {"", "none"}
        or str(row.get("status", "")).strip().upper() not in {"", "PASS"}
    ]
    _check_condition(
        rows,
        "secret_scrub_zero_findings",
        not finding_rows,
        str(path),
        "zero findings" if not finding_rows else f"{len(finding_rows)} finding rows",
    )


def _check_optional_experiment_artifacts(rows: list[dict[str, Any]], tables_dir: Path, figures_dir: Path) -> None:
    table_text = ""
    for path in tables_dir.glob("paper_table*.csv"):
        try:
            table_text += path.read_text(encoding="utf-8")
        except Exception:
            continue
    if "paper_repeatability_small" in table_text:
        _check_path(rows, "optional_repeatability_table", tables_dir / "paper_table10_repeatability.csv", nonempty=True)
        for suffix in (".png", ".svg"):
            _check_path(rows, "optional_repeatability_figure", figures_dir / f"appendix_repeatability_variance{suffix}", nonempty=True)
    if "paper_livecodebench_fresh_small" in table_text:
        _check_path(rows, "optional_fresh_coding_table", tables_dir / "paper_table11_fresh_coding.csv", nonempty=True)
        for suffix in (".png", ".svg"):
            _check_path(rows, "optional_fresh_coding_figure", figures_dir / f"appendix_fresh_coding{suffix}", nonempty=True)


def _check_croissant_metadata(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    allow_placeholder_url: bool,
    strict_final_submission: bool,
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        _check_condition(rows, "croissant_required_fields", False, str(path), "missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _check_condition(rows, "croissant_required_fields", False, str(path), f"invalid JSON: {exc}")
        return
    missing = [field for field in REQUIRED_CROISSANT_FIELDS if field not in payload]
    distribution_ok = isinstance(payload.get("distribution"), list) and bool(payload.get("distribution"))
    record_set_ok = isinstance(payload.get("recordSet"), list) and bool(payload.get("recordSet"))
    required_ok = not missing and distribution_ok and record_set_ok
    _check_condition(
        rows,
        "croissant_required_fields",
        required_ok,
        str(path),
        "present" if required_ok else f"missing={missing}; distribution_ok={distribution_ok}; recordSet_ok={record_set_ok}",
    )
    url = str(payload.get("url") or "")
    placeholder_ok = allow_placeholder_url or url != PLACEHOLDER_DATASET_URL
    _check_condition(
        rows,
        "croissant_url",
        placeholder_ok,
        str(path),
        "placeholder allowed" if allow_placeholder_url and url == PLACEHOLDER_DATASET_URL else ("non-placeholder" if placeholder_ok else "placeholder URL"),
    )
    distributions = payload.get("distribution") if isinstance(payload.get("distribution"), list) else []
    urls = [str(item.get("contentUrl") or "") for item in distributions if isinstance(item, dict)]
    hosted_count = sum(1 for url_value in urls if url_value.startswith(("https://", "http://")))
    release_mode = str(payload.get("releaseMode") or "")
    release_mode_description = str(payload.get("releaseModeDescription") or "")
    relative_mode_ok = release_mode == "relative_path_archive" and bool(release_mode_description.strip())
    _check_condition(
        rows,
        "croissant_release_mode",
        bool(hosted_count == len(urls) and urls) or relative_mode_ok,
        str(path),
        f"hosted_urls={hosted_count}/{len(urls)}; releaseMode={release_mode or 'missing'}",
    )
    if strict_final_submission:
        _check_condition(
            rows,
            "strict_hosted_distribution_urls",
            bool(urls) and hosted_count == len(urls),
            str(path),
            f"hosted_urls={hosted_count}/{len(urls)}",
        )


def _check_main_track_artifacts(rows: list[dict[str, Any]], artifact_root: Path) -> None:
    expected = _main_suites_from_tasks()
    artifact_suites = _artifact_suites(artifact_root)
    missing = sorted(expected - artifact_suites)
    _check_condition(
        rows,
        "main_track_artifacts",
        not missing,
        str(artifact_root),
        "all main suites have artifacts" if not missing else f"missing={missing}",
    )


def _check_official_placeholders_not_main(rows: list[dict[str, Any]], artifact_root: Path) -> None:
    bad: list[str] = []
    for outcomes_path in artifact_root.glob("*/outcomes.jsonl"):
        config = _load_config(outcomes_path.parent / "config_snapshot.yaml")
        role = str((config.get("metadata") or {}).get("paper_role") or "")
        if role not in {"main", "paper"}:
            continue
        try:
            outcome_rows = _read_jsonl(outcomes_path)
        except Exception:
            continue
        if any(_placeholder_unverified(row) for row in outcome_rows):
            bad.append(str(outcomes_path))
    _check_condition(
        rows,
        "main_metrics_no_official_placeholders",
        not bad,
        str(artifact_root),
        "no main placeholder labels" if not bad else f"placeholder outcomes in {bad[:3]}",
    )


def _check_incompatible_main_rows(rows: list[dict[str, Any]]) -> None:
    bad: list[str] = []
    for root in (Path("data/processed"), Path("data/tasks")):
        if not root.exists():
            continue
        for path in sorted(root.glob("*.jsonl")):
            try:
                records = _read_jsonl(path)
            except Exception:
                continue
            for index, record in enumerate(records, start=1):
                if _is_bad_incompatible_main_row(record):
                    bad.append(f"{path}:{index}")
                    break
    for path in sorted(Path("reports/tables").glob("*.csv")):
        for index, record in enumerate(_read_csv(path), start=2):
            if _is_bad_incompatible_main_row(record):
                bad.append(f"{path}:{index}")
                break
    _check_condition(
        rows,
        "provider_incompatible_tracks_not_main",
        not bad,
        "data/processed,data/tasks,reports/tables",
        "no incompatible main rows" if not bad else f"incompatible main rows in {bad[:5]}",
    )


def _is_bad_incompatible_main_row(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    role = str(record.get("paper_role") or metadata.get("paper_role") or "").strip().lower()
    if role not in {"paper", "main", "main_text"}:
        return False
    text = json.dumps(record, sort_keys=True).lower()
    if not any(pattern in text for pattern in INCOMPATIBLE_MAIN_TRACK_PATTERNS):
        return False
    status = str(record.get("official_harness_status") or metadata.get("official_harness_status") or "").strip().lower()
    compatible = _truthy(record.get("chat_completion_compatible") or metadata.get("chat_completion_compatible"))
    return not (status == "completed" and compatible)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _check_fresh_coding_claim(rows: list[dict[str, Any]], tables_dir: Path) -> None:
    table = tables_dir / "paper_table11_fresh_coding.csv"
    if not table.exists() or table.stat().st_size == 0:
        _check_condition(rows, "fresh_coding_official_claim", True, str(table), "no fresh coding claim")
        return
    with table.open(encoding="utf-8", newline="") as f:
        table_rows = list(csv.DictReader(f))
    completed = [row for row in table_rows if row.get("official_harness_status") == "completed"]
    if not completed:
        _check_condition(rows, "fresh_coding_official_claim", True, str(table), "fresh coding not claimed completed")
        return
    official_root = Path("reports/artifacts_livecodebench_official")
    official_rows = []
    for outcomes_path in official_root.glob("*/outcomes.jsonl"):
        official_rows.extend(_read_jsonl(outcomes_path))
    ok = bool(official_rows) and all(
        (row.get("metadata") or {}).get("label_source") == "official_livecodebench" for row in official_rows
    )
    _check_condition(
        rows,
        "fresh_coding_official_claim",
        ok,
        str(table),
        "official labels found" if ok else "completed claim without official_livecodebench labels",
    )


def _check_replacement_benchmark_outputs(rows: list[dict[str, Any]], tables_dir: Path, figures_dir: Path) -> None:
    active = (
        Path("configs/paper_bigcodebench_hard.yaml").exists()
        or Path("configs/paper_canitedit_descriptive.yaml").exists()
        or (tables_dir / "paper_table18_bigcodebench_hard.csv").exists()
        or (tables_dir / "paper_table19_canitedit_descriptive.csv").exists()
    )
    if not active:
        return
    required_tables = [
        "paper_table18_bigcodebench_hard.csv",
        "paper_table19_canitedit_descriptive.csv",
        "paper_table20_replacement_token_usage_proxy.csv",
        "paper_table21_replacement_fixed_budget_scheduling.csv",
    ]
    for table in required_tables:
        _check_path(rows, "replacement_required_table", tables_dir / table, nonempty=True)
    for prefix in [
        "paper_figure11_bigcodebench_hard_success_by_budget",
        "paper_figure12_canitedit_descriptive_success_by_budget",
        "paper_figure13_replacement_token_proxy_vs_success",
        "paper_figure14_replacement_allocation_frontier",
    ]:
        for suffix in (".png", ".svg"):
            _check_path(rows, "replacement_required_figure", figures_dir / f"{prefix}{suffix}", nonempty=True)

    table18 = _read_csv(tables_dir / "paper_table18_bigcodebench_hard.csv")
    statuses = {row.get("official_harness_status", "") for row in table18}
    allowed_statuses = {
        "completed",
        "official_labels_completed",
        "official_labels_absent",
        "configured_not_run",
        "not_run",
        "official_harness_unavailable",
    }
    _check_condition(
        rows,
        "replacement_bigcodebench_official_status",
        bool(table18) and statuses <= allowed_statuses and "" not in statuses,
        str(tables_dir / "paper_table18_bigcodebench_hard.csv"),
        f"statuses={sorted(statuses)}",
    )

    table19 = _read_csv(tables_dir / "paper_table19_canitedit_descriptive.csv")
    verifier_statuses = {row.get("verifier_status", "") for row in table19}
    _check_condition(
        rows,
        "replacement_canitedit_pass_fail_labels",
        bool(table19) and "" not in verifier_statuses and "missing_labels" not in verifier_statuses,
        str(tables_dir / "paper_table19_canitedit_descriptive.csv"),
        f"statuses={sorted(verifier_statuses)}",
    )

    table20 = _read_csv(tables_dir / "paper_table20_replacement_token_usage_proxy.csv")
    coverage_ok = bool(table20) and all("usage_forecast_coverage" in row for row in table20)
    _check_condition(
        rows,
        "replacement_token_proxy_coverage_reported",
        coverage_ok,
        str(tables_dir / "paper_table20_replacement_token_usage_proxy.csv"),
        "coverage column present" if coverage_ok else "missing coverage rows",
    )

    table21 = _read_csv(tables_dir / "paper_table21_replacement_fixed_budget_scheduling.csv")
    policies = {row.get("policy", "") for row in table21}
    non_oracle = {policy for policy in policies if policy and policy != "oracle"}
    _check_condition(
        rows,
        "replacement_fixed_budget_policies",
        bool(table21) and "oracle" in policies and len(non_oracle) >= 3,
        str(tables_dir / "paper_table21_replacement_fixed_budget_scheduling.csv"),
        f"policies={sorted(policies)}",
    )

    swe_table = _read_csv(tables_dir / "appendix_swe_official_mini.csv")
    swe_ok = not swe_table or all(
        "appendix" in str(row.get("notes", "")).lower() or row.get("official_harness_status") == "completed"
        for row in swe_table
    )
    _check_condition(
        rows,
        "replacement_swe_appendix_only",
        swe_ok,
        str(tables_dir / "appendix_swe_official_mini.csv"),
        "SWE mini is appendix/infrastructure unless completed",
    )


def _check_clean_evidence_scope(rows: list[dict[str, Any]], path: Path) -> None:
    table_rows = _read_csv(path)
    expected_sources = {
        "GSM8K + MATH",
        "HumanEval+ + MBPP+",
        "BigCodeBench-Hard",
        "CanItEdit",
        "LiveCodeBench-300",
        "prompt variants",
    }
    sources = {row.get("source", "") for row in table_rows}
    text = "\n".join(json.dumps(row, sort_keys=True).lower() for row in table_rows)
    forbidden = ["mock-model", "official_labels_absent", "smoke", "docker", "openhands", "aider"]
    bad_terms = [term for term in forbidden if term in text]
    _check_condition(
        rows,
        "clean_evidence_scope",
        len(table_rows) == 6 and expected_sources <= sources and not bad_terms,
        str(path),
        "six clean paper-facing rows" if len(table_rows) == 6 and expected_sources <= sources and not bad_terms else f"rows={len(table_rows)} missing={sorted(expected_sources - sources)} bad_terms={bad_terms}",
    )


def _check_strict_fixed_budget_tables(rows: list[dict[str, Any]], tables_dir: Path) -> None:
    for name in STRICT_FIXED_BUDGET_TABLES:
        path = tables_dir / name
        if not path.exists() or path.stat().st_size == 0:
            _check_condition(rows, "strict_fixed_budget_scheduling", True, str(path), "not present")
            continue
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                table_rows = list(reader)
        except Exception as exc:
            _check_condition(rows, "strict_fixed_budget_scheduling", False, str(path), f"invalid CSV: {exc}")
            continue

        missing = sorted(set(STRICT_FIXED_BUDGET_COLUMNS) - fieldnames)
        if missing:
            _check_condition(rows, "strict_fixed_budget_scheduling", False, str(path), f"missing_columns={missing}")
            continue
        if not table_rows:
            _check_condition(rows, "strict_fixed_budget_scheduling", True, str(path), "no rows; strict columns present")
            continue

        violations: list[str] = []
        for index, row in enumerate(table_rows, start=2):
            if _overshoot_allowed(row):
                continue
            target_total_budget = _csv_int(row.get("target_total_budget"))
            selected_total_budget = _csv_int(row.get("selected_total_budget"))
            budget_used = _csv_int(row.get("budget_used"))
            budget_slack_tokens = _csv_int(row.get("budget_slack_tokens"))
            feasible = (
                selected_total_budget <= target_total_budget
                and budget_used <= selected_total_budget
                and budget_used <= target_total_budget
                and budget_slack_tokens == target_total_budget - budget_used
                and budget_slack_tokens >= 0
                and _truthy(row.get("strict_budget_feasible"))
            )
            if not feasible:
                violations.append(
                    f"line {index}: target={target_total_budget} "
                    f"selected={selected_total_budget} used={budget_used} slack={budget_slack_tokens} "
                    f"strict_budget_feasible={row.get('strict_budget_feasible')}"
                )
        _check_condition(
            rows,
            "strict_fixed_budget_scheduling",
            not violations,
            str(path),
            "strict rows feasible" if not violations else "; ".join(violations[:5]),
        )


def _overshoot_allowed(row: dict[str, Any]) -> bool:
    for key in ("mode", "selection_mode", "budget_selection_mode"):
        if str(row.get(key) or "").strip().lower() == "overshoot_allowed":
            return True
    return False


def _csv_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _check_report_scope(rows: list[dict[str, Any]], path: Path) -> None:
    if not path.exists():
        _check_condition(rows, "paper_report_scope", False, str(path), "missing")
        return
    text = path.read_text(encoding="utf-8").lower()
    required = ["evidence scope", "math", "coding", "swe", "not"]
    missing = [item for item in required if item not in text]
    _check_condition(
        rows,
        "paper_report_scope",
        not missing,
        str(path),
        "mentions exact evidence scope" if not missing else f"missing_terms={missing}",
    )


def _main_suites_from_tasks() -> set[str]:
    suites: set[str] = set()
    for task_path in Path("data/processed").glob("paper_*.jsonl"):
        try:
            task_rows = _read_jsonl(task_path)
        except Exception:
            continue
        if any(str((row.get("metadata") or {}).get("paper_role") or "") in {"main", "paper"} for row in task_rows):
            suites.add(task_path.stem)
    return suites


def _artifact_suites(artifact_root: Path) -> set[str]:
    suites: set[str] = set()
    for config_path in artifact_root.glob("*/config_snapshot.yaml"):
        config = _load_config(config_path)
        suite = str(config.get("suite") or config.get("suite_name") or (config.get("metadata") or {}).get("suite_name") or "")
        if suite:
            suites.add(suite)
            continue
        name = config_path.parent.name
        if "__" in name:
            suites.add(name.split("__", maxsplit=1)[0])
    return suites


def _placeholder_unverified(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    verification_metadata = verification.get("metadata") if isinstance(verification.get("metadata"), dict) else {}
    details = verification.get("details") if isinstance(verification.get("details"), dict) else {}
    return (
        metadata.get("label_source") == "official_harness_placeholder"
        or metadata.get("exclude_from_main_metrics") is True
        or verification_metadata.get("label_source") == "official_harness_placeholder"
        or verification_metadata.get("exclude_from_main_metrics") is True
        or details.get("error") == "official_harness_required"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "path", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TokenCapBench submission package outputs.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--docs-dir", default="docs", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--output", default="reports/tables/submission_package_validation.csv")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--allow-placeholder-url", action="store_true")
    parser.add_argument("--strict-final-submission", action="store_true")
    args = parser.parse_args()
    path, ok = validate_submission_package(
        artifact_root=args.artifact_root,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        docs_dir=args.docs_dir,
        metadata_dir=args.metadata_dir,
        output=args.output,
        run_pytest=args.run_pytest,
        allow_placeholder_url=args.allow_placeholder_url,
        strict_final_submission=args.strict_final_submission,
    )
    print(f"Wrote {path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
